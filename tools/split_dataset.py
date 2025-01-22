import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm  # 导入 tqdm 库


def split_one_image(image_path, label_path, output_image_dir, output_label_dir):
    # 读取原始图片
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    image_name = Path(image_path).name

    # 定义子图片的尺寸
    tile_width = 1280
    tile_height = 1280

    # 读取标注文件
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])

            # 转换为绝对坐标
            x_center = x_center_norm * image_width
            y_center = y_center_norm * image_height
            width = width_norm * image_width
            height = height_norm * image_height

            x_min = x_center - width / 2
            x_max = x_center + width / 2
            y_min = y_center - height / 2
            y_max = y_center + height / 2

            labels.append({
                'class_id': class_id,
                'x_min': x_min,
                'x_max': x_max,
                'y_min': y_min,
                'y_max': y_max
            })

    # 计算子图片的起始坐标（确保覆盖整个原始图片）
    x_steps = list(range(0, image_width, tile_width))
    if x_steps[-1] + tile_width < image_width:
        x_steps.append(image_width - tile_width)

    y_steps = list(range(0, image_height, tile_height))
    if y_steps[-1] + tile_height < image_height:
        y_steps.append(image_height - tile_height)

    # 对每个子图片进行处理
    tile_id = 0
    for y_min_tile in y_steps:
        for x_min_tile in x_steps:
            x_max_tile = x_min_tile + tile_width
            y_max_tile = y_min_tile + tile_height

            # 裁剪子图片并保存
            tile_image = image[y_min_tile:y_max_tile, x_min_tile:x_max_tile]
            tile_image_name = f'{image_name}__{tile_id}.jpg'
            cv2.imwrite(os.path.join(output_image_dir, tile_image_name), tile_image)

            # 处理标注
            tile_labels = []
            for label in labels:
                # 检查标注框是否与子图片有重叠
                x_min_overlap = max(label['x_min'], x_min_tile)
                x_max_overlap = min(label['x_max'], x_max_tile)
                y_min_overlap = max(label['y_min'], y_min_tile)
                y_max_overlap = min(label['y_max'], y_max_tile)

                if x_min_overlap < x_max_overlap and y_min_overlap < y_max_overlap:
                    # 计算相对于子图片的坐标
                    x_center_tile = ((x_min_overlap + x_max_overlap) / 2) - x_min_tile
                    y_center_tile = ((y_min_overlap + y_max_overlap) / 2) - y_min_tile
                    width_tile = x_max_overlap - x_min_overlap
                    height_tile = y_max_overlap - y_min_overlap

                    # 归一化坐标
                    x_center_norm = x_center_tile / tile_width
                    y_center_norm = y_center_tile / tile_height
                    width_norm = width_tile / tile_width
                    height_norm = height_tile / tile_height

                    # 添加到子图片的标注列表
                    tile_labels.append(f"{label['class_id']} {x_center_norm} {y_center_norm} {width_norm} {height_norm}")

            # 保存子图片的标注文件
            tile_label_name = f'{image_name}__{tile_id}.txt'
            with open(os.path.join(output_label_dir, tile_label_name), 'w') as f:
                f.write('\n'.join(tile_labels))

            tile_id += 1


def main():
    # 原始图片的路径和标注文件的路径
    image_dir = Path("/home/liuyang/datasets/20240922_frames_label/images")
    label_dir = Path("/home/liuyang/datasets/20240922_frames_label/labels")

    # 创建输出文件夹
    output_image_dir = Path("/home/liuyang/datasets/20240922_split-1280/images")
    output_label_dir = Path("/home/liuyang/datasets/20240922_split-1280/labels")
    output_image_dir.mkdir(exist_ok=False)
    output_label_dir.mkdir(exist_ok=False)

    # 获取图片列表
    image_list = list(image_dir.iterdir())

    # 使用 tqdm 显示进度条
    for img in tqdm(image_list, desc="Processing images"):
        img_stem = img.stem
        label_name = f"{img_stem}.txt"
        label_path = label_dir / label_name
        if label_path.is_file():
            split_one_image(str(img), str(label_path), str(output_image_dir), str(output_label_dir))


if __name__ == "__main__":
    main()


