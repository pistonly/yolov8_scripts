import cv2
import numpy as np
from pathlib import Path
from ultralytics.utils.instance import Instances

def draw_bboxes(img, bboxes, classes, labels=None, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制边界框

    Args:
        img (np.ndarray): 输入图像 (H, W, C)
        bboxes (np.ndarray): 边界框坐标，形状为 (N, 4)，格式为 [x1, y1, x2, y2]
        classes (np.ndarray): 每个框的类别索引，形状为 (N,)
        labels (List[str] | None): 类别名称映射表，可选
        color (Tuple[int, int, int]): 绘制框的颜色 (B, G, R)
        thickness (int): 框线条的厚度

    Returns:
        np.ndarray: 带有绘制的图像
    """
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2 = map(int, box)
        class_id = classes[i]
        label = labels[class_id] if labels else str(class_id)

        # 画框
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 在框上方画类别
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        text_x, text_y = x1, y1 - 5
        if text_y < 0:  # 防止文字超出图像顶部
            text_y = y1 + text_size[1] + 5

        cv2.rectangle(img, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0], text_y), color, -1)  # 文本背景
        cv2.putText(img, label, (text_x, text_y - 2), font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

    return img

def visualize_augmented_data(image_dir, label_dir, labels=None):
    """
    可视化增广结果（包括图片和标注框）

    Args:
        image_dir (Path): 图像文件夹路径
        label_dir (Path): YOLO 格式标签文件夹路径
        labels (List[str] | None): 类别名称映射表，可选

    Returns:
        None
    """
    image_paths = sorted(image_dir.glob("*.*"))  # 遍历图像文件夹

    for img_path in image_paths:
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Failed to load image {img_path}")
            continue

        h, w = img.shape[:2]

        # 读取对应的标注文件
        label_path = label_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"Warning: Label file not found for {img_path}")
            continue

        with open(label_path, "r") as f:
            lines = f.readlines()

        bboxes = []
        classes = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            c, x, y, bw, bh = map(float, line.split())
            c = int(c)
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            bboxes.append([x1, y1, x2, y2])
            classes.append(c)

        bboxes = np.array(bboxes, dtype=np.float32)
        classes = np.array(classes, dtype=np.int32)

        # 在图像上绘制标注框
        img_with_bboxes = draw_bboxes(img.copy(), bboxes, classes, labels=labels)

        # 显示结果
        cv2.imshow("Augmented Image", img_with_bboxes)

        # 等待按键（按 q 退出）
        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 输入增广后的图像和标签目录
    dataset_dir = Path("/home/liuyang/datasets/xfd_augument_half_new/20241214151008180-11-1-main_yolo_augument")
    # dataset_dir = Path("/home/liuyang/datasets/als_uav_ds_augment/20241216101114823-11-1-main_yolo/")
    augmented_image_dir = dataset_dir / "images"
    augmented_label_dir = dataset_dir / "labels"

    # 类别名称映射表（可选）
    label_names = ["class0", "class1", "class2", "class3", "class4"]  # 根据实际类别填写

    # 调用可视化函数
    visualize_augmented_data(augmented_image_dir, augmented_label_dir, labels=label_names)
