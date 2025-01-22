import cv2
import numpy as np
from pathlib import Path

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



def process_image_and_label(source_image_path, source_label_path, target_image_path, target_label_path):
    """
    处理单张图像和对应的 YOLO 格式标注文件，
    将图像缩小至一半并放置在中心，同时调整标注坐标。

    Args:
        source_image_path (Path): 原始图像路径
        source_label_path (Path): 原始标注路径
        target_image_path (Path): 目标图像保存路径
        target_label_path (Path): 目标标注保存路径
    """
    # 读取图像
    img = cv2.imread(str(source_image_path))
    img_h, img_w = img.shape[:2]

    # 创建空白画布
    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # 缩小图像到一半
    resized_img = cv2.resize(img, (img_w // 2, img_h // 2))

    # 将缩小后的图像放置到画布中心
    start_x = img_w // 4
    start_y = img_h // 4
    canvas[start_y:start_y + img_h // 2, start_x:start_x + img_w // 2] = resized_img

    # 保存处理后的图像
    target_image_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target_image_path), canvas)

    # 处理 YOLO 标注
    cls_array, bboxes_xyxy = read_yolo_label(source_label_path, img_w, img_h)

    # 调整标注坐标以适配新图像
    bboxes_xyxy[:, [0, 2]] = bboxes_xyxy[:, [0, 2]] / 2 + start_x
    bboxes_xyxy[:, [1, 3]] = bboxes_xyxy[:, [1, 3]] / 2 + start_y

    # 保存调整后的标注
    write_yolo_label(target_label_path, cls_array, bboxes_xyxy, img_w, img_h)

# 数据集路径
source_dataset = Path("/home/liuyang/datasets/xfd_augument")
target_dataset = Path("/home/liuyang/datasets/xfd_augument_half_new")

# 获取所有图像路径
image_paths = sorted(source_dataset.glob("**/*.jpg"))

for source_image_path in image_paths:
    # 对应的标注路径

    source_label_path = Path(str(source_image_path.with_suffix('.txt')).replace("/images", "/labels"))

    if not source_label_path.exists():
        print(f"Warning: Label file not found for {source_image_path}")
        continue

    # 目标路径
    relative_path = source_image_path.relative_to(source_dataset)
    target_image_path = target_dataset / relative_path

    relative_label_path = source_label_path.relative_to(source_dataset)
    target_label_path = target_dataset / relative_label_path

    # 处理图像和标注
    process_image_and_label(source_image_path, source_label_path, target_image_path, target_label_path)

print("Dataset processing complete.")
