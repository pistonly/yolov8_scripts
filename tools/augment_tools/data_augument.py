import os
import random
import cv2
import numpy as np
from pathlib import Path
from copy import deepcopy
from augment import RandomPerspective, RandomFlip, RandomHSV, Instances


def crop_with_first_instance(img, bboxes, crop_size=(1280, 1280)):
    """
    Crop a region from the input image based on the first bounding box.

    Args:
        img (np.ndarray): Input image (H, W, C).
        bboxes (np.ndarray): Bounding boxes in xyxy format with shape (N, 4).
        crop_size (Tuple[int, int]): Crop size (width, height).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Cropped image and updated bounding boxes.
    """
    h, w = img.shape[:2]
    crop_w, crop_h = crop_size

    # Ensure the crop size is smaller than the image
    if crop_w > w or crop_h > h:
        raise ValueError(
            "Crop size must be smaller than the image dimensions.")

    # Use the first bounding box as the center of the crop
    if len(bboxes) == 0:
        raise ValueError("No bounding boxes available for cropping.")

    # Get the first bounding box
    x1, y1, x2, y2 = bboxes[0]
    topleft_x_min = int(max(0, x2 - crop_w))
    topleft_y_min = int(max(0, y2 - crop_h))
    topleft_x_max = int(min(x1, w - crop_w))
    topleft_y_max = int(min(y1, h - crop_h))

    # Compute the crop's top-left corner
    crop_x_min = random.randint(topleft_x_min, topleft_x_max)
    crop_y_min = random.randint(topleft_y_min, topleft_y_max)

    # Ensure the crop stays within the image bounds
    crop_x_min = min(crop_x_min, w - crop_w)
    crop_y_min = min(crop_y_min, h - crop_h)

    crop_x_max = crop_x_min + crop_w
    crop_y_max = crop_y_min + crop_h

    # Crop the image
    cropped_img = img[int(crop_y_min):int(crop_y_max),
                      int(crop_x_min):int(crop_x_max)]

    # Adjust bounding boxes
    updated_bboxes = []
    for bbox in bboxes:
        bx_min, by_min, bx_max, by_max = bbox

        # Clip bounding boxes to the crop area
        clipped_x1 = max(0, bx_min - crop_x_min)
        clipped_y1 = max(0, by_min - crop_y_min)
        clipped_x2 = min(crop_w, bx_max - crop_x_min)
        clipped_y2 = min(crop_h, by_max - crop_y_min)

        # Check if the bounding box is valid within the crop
        if clipped_x1 < clipped_x2 and clipped_y1 < clipped_y2:
            updated_bboxes.append(
                [clipped_x1, clipped_y1, clipped_x2, clipped_y2])

    return cropped_img, np.array(updated_bboxes, dtype=np.float32)


def read_yolo_label(label_path, img_w, img_h):
    """
    读取 YOLO 标注文件，并转换为 [x1, y1, x2, y2] 格式

    Args:
        label_path (Path): YOLO 标注文件路径
        img_w (int): 图像宽度
        img_h (int): 图像高度

    Returns:
        cls (np.ndarray): 类别索引数组 (N,)
        bboxes_xyxy (np.ndarray): Bounding box in xyxy format (N, 4)
    """
    with open(label_path, 'r') as f:
        lines = f.readlines()

    cls_list = []
    bboxes_xyxy_list = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        c, x, y, w, h = line.split()
        c = int(c)
        x, y, w, h = map(float, [x, y, w, h])
        # YOLO 格式: x_center, y_center, width, height (归一化后)
        # 转换为 xyxy
        # 先将归一化值变为绝对坐标
        xc = x * img_w
        yc = y * img_h
        bw = w * img_w
        bh = h * img_h

        x1 = xc - bw / 2
        y1 = yc - bh / 2
        x2 = xc + bw / 2
        y2 = yc + bh / 2

        cls_list.append(c)
        bboxes_xyxy_list.append([x1, y1, x2, y2])

    cls_array = np.array(cls_list, dtype=np.int32)
    bboxes_xyxy = np.array(bboxes_xyxy_list, dtype=np.float32)

    return cls_array, bboxes_xyxy


def write_yolo_label(save_label_path, cls_array, bboxes_xyxy, img_w, img_h):
    """
    将 xyxy 格式的标注写回 YOLO 格式，并保存到文件

    Args:
        save_label_path (Path): 要保存的 YOLO 标注文件路径
        cls_array (np.ndarray): (N,) 类别数组
        bboxes_xyxy (np.ndarray): (N, 4) xyxy 格式
        img_w (int): 图像宽度
        img_h (int): 图像高度
    """
    save_label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for c, bbox in zip(cls_array, bboxes_xyxy):
        x1, y1, x2, y2 = bbox
        # 先确保数值在图像范围内（clip），防止越界
        x1 = max(0, min(x1, img_w - 1))
        x2 = max(0, min(x2, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        y2 = max(0, min(y2, img_h - 1))

        bw = x2 - x1
        bh = y2 - y1
        xc = x1 + bw / 2
        yc = y1 + bh / 2

        # 再转成归一化的 YOLO 格式
        x_normalized = xc / img_w
        y_normalized = yc / img_h
        w_normalized = bw / img_w
        h_normalized = bh / img_h

        # 组装行字符串：class x_center y_center w h
        line_str = f"{c} {x_normalized:.6f} {y_normalized:.6f} {w_normalized:.6f} {h_normalized:.6f}"
        lines.append(line_str)

    with open(save_label_path, 'w') as f:
        for l in lines:
            f.write(f"{l}\n")


def create_instances(bboxes_xyxy, img_shape, segments=None, keypoints=None):
    """
    将 bboxes, segments, keypoints 封装为 ultralytics.utils.instance.Instances 对象

    Args:
        bboxes_xyxy (np.ndarray): (N, 4) xyxy 格式的 bbox
        img_shape (Tuple[int, int]): (H, W) 图像形状
        segments (List[np.ndarray] | None): 分割多边形列表
        keypoints (np.ndarray | None): 关键点 (N, K, 3)

    Returns:
        instances (Instances): ultralytics.utils.instance.Instances 对象
    """
    if segments is None:
        segments = []
    # segments: List[np.ndarray], shape of each [num_points, 2]
    # keypoints: np.ndarray shape (N, K, 3)
    return Instances(bboxes=bboxes_xyxy,
                     segments=segments,
                     keypoints=keypoints,
                     bbox_format="xyxy",
                     normalized=False)


def get_image_and_label_paths(source):
    """
    遍历文件夹下所有 jpg 图片，或从指定的文本文件中获取图片路径，并找到对应的标签文件。

    Args:
        source (Path): 数据集根目录或包含图片路径的文本文件

    Returns:
        List[Path], List[Path]: 图片路径列表和标签路径列表
    """
    if source.is_file():
        # 如果是文件，则读取文件中的图片路径
        base_path = source.parent
        with open(source, 'r') as f:
            img_paths = [
                base_path / line.strip() for line in f if line.strip()
            ]
    else:
        # 如果是文件夹，则遍历文件夹中的所有 jpg 文件
        img_paths = sorted(source.glob("**/*.jpg"))

    label_paths = [
        Path(
            str(img_path).replace("/images/",
                                  "/labels/").replace(".jpg", ".txt"))
        for img_path in img_paths
    ]

    return img_paths, label_paths


def load_image_and_labels(img_path, label_path):
    """
    加载图像和对应的标签。

    Args:
        img_path (Path): 图片路径
        label_path (Path): 标签路径

    Returns:
        np.ndarray, np.ndarray, np.ndarray: 图像, 类别数组, 边界框数组
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    h, w = img.shape[:2]

    if label_path.exists():
        cls_array, bboxes_xyxy = read_yolo_label(label_path, w, h)
    else:
        cls_array, bboxes_xyxy = np.array([]), np.array([])

    return img, cls_array, bboxes_xyxy


def augment_image(img, cls_array, bboxes_xyxy, transform, flip_transform,
                  hsv_transform):
    """
    对图像及其标签进行增强。

    Args:
        img (np.ndarray): 输入图像
        cls_array (np.ndarray): 类别数组
        bboxes_xyxy (np.ndarray): 边界框数组
        transform: 随机透视变换对象
        flip_transform: 随机翻转变换对象
        hsv_transform: HSV 变换对象

    Returns:
        np.ndarray, np.ndarray, np.ndarray: 增强后的图像, 类别数组, 边界框数组
    """
    instances = create_instances(bboxes_xyxy, img.shape[:2])
    labels_dict = {"img": img, "cls": cls_array, "instances": instances}

    # 随机透视变换
    augmented = transform(deepcopy(labels_dict))
    aug_img = augmented["img"]
    aug_instances = augmented["instances"]
    aug_cls = augmented["cls"]

    if len(aug_instances.bboxes) == 0:
        return None, None, None

    final_bboxes_xyxy = aug_instances.bboxes
    cropped_img, cropped_bboxes = crop_with_first_instance(
        aug_img, final_bboxes_xyxy)

    # 随机翻转
    label_dict_cropped = {
        "img": cropped_img,
        "instances": create_instances(cropped_bboxes, cropped_img.shape[:2])
    }
    flip_result = flip_transform(label_dict_cropped)
    fliped_img = flip_result["img"]
    fliped_instances = flip_result["instances"]
    fliped_instances.convert_bbox(format="xyxy")

    # HSV 变换
    fliped_img = hsv_transform({"img": fliped_img})["img"]

    return fliped_img, aug_cls, fliped_instances.bboxes


def save_augmented_data(save_dir, img_relative_path, idx, img, cls_array, bboxes_xyxy):
    """
    保存增强后的图像和标签。

    Args:
        save_dir (Path): 保存目录
        img_path (Path): 原始图片路径
        idx (int): 增强编号
        img (np.ndarray): 增强后的图像
        cls_array (np.ndarray): 类别数组
        bboxes_xyxy (np.ndarray): 边界框数组
    """
    save_image_dir = save_dir / img_relative_path.parent
    save_label_dir = Path(str(save_image_dir).replace("images", "labels"))
    save_image_dir.mkdir(parents=True, exist_ok=True)
    save_label_dir.mkdir(parents=True, exist_ok=True)

    save_image_name = f"{img_relative_path.stem}_aug{idx}.jpg"
    save_label_name = f"{img_relative_path.stem}_aug{idx}.txt"

    save_img_path = save_image_dir / save_image_name
    save_label_path = save_label_dir / save_label_name

    try:
        cv2.imwrite(str(save_img_path), img)
    except:
        return

    write_yolo_label(save_label_path, cls_array, bboxes_xyxy, img.shape[1],
                     img.shape[0])


def main(augment_times=7):
    # source = Path("/home/liuyang/datasets/als_uav_ds/image_list.txt")
    # target_dir = Path("/home/liuyang/datasets/als_uav_ds_augment/")
    source = Path("/home/liuyang/datasets/als_uav_ds/20241216102315093-20-2-main_yolo/")
    target_dir = Path("tmp/")
    source_folder = source.parent if source.is_file() else source

    transform = RandomPerspective(degrees=10.0,
                                  translate=0.1,
                                  scale=0.5,
                                  shear=10.0,
                                  perspective=0.0,
                                  border=(0, 0))
    flip_transform = RandomFlip()
    hsv_transform = RandomHSV(hgain=0.015, sgain=0.7, vgain=0.4)

    img_paths, label_paths = get_image_and_label_paths(source)

    for img_path, label_path in zip(img_paths, label_paths):
        try:
            img, cls_array, bboxes_xyxy = load_image_and_labels(
                img_path, label_path)
        except ValueError as e:
            print(e)
            continue

        for idx in range(augment_times):
            augmented_data = augment_image(img, cls_array, bboxes_xyxy,
                                           transform, flip_transform,
                                           hsv_transform)
            if augmented_data is None:
                continue

            aug_img, aug_cls, aug_bboxes = augmented_data

            img_relative_path = img_path.relative_to(source_folder)
            save_augmented_data(target_dir, img_relative_path, idx, aug_img, aug_cls,
                                aug_bboxes)

    print("Data augmentation finished!")


if __name__ == "__main__":
    main()
