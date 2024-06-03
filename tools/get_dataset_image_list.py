import glob
import os
import numpy as np
from pathlib import Path

def collect_jpg_files_glob(directory_path):
    # 构建查找.jpg文件的路径模式
    pattern = os.path.join(directory_path, '**', '*.jpg')

    # 使用glob.glob()查找所有匹配的文件，recursive=True允许搜索子目录
    jpg_files = glob.glob(pattern, recursive=True)

    return jpg_files

# 指定要遍历的文件夹路径
directory_path = '/home/liuyang/datasets/PANDA/YOLO/train_baby_carriage/'

# 调用函数并打印结果
img_files = collect_jpg_files_glob(directory_path)
img_files = [Path(f).relative_to(directory_path) for f in img_files]

image_list = Path(directory_path) / "image_list.txt"
if image_list.is_file():
    raise RuntimeError(f"{str(image_list)} exist!")
else:
    with open(str(image_list), "w") as f:
        for img_p in img_files:
            f.write(f"./{str(img_p)}")
            f.write("\n")

