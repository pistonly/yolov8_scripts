import glob
import os
import numpy as np
from pathlib import Path
import random 

directory_path = "/home/liuyang/datasets/als_xfd-half_uav_augment/"
relative_to = "/home/liuyang/datasets/als_xfd-half_uav_augment/"
image_list = "/home/liuyang/datasets/als_xfd-half_uav_augment/image_list.txt"

exclude = "exclude"


def collect_jpg_files_glob(directory_path):
    # 构建查找.jpg文件的路径模式
    pattern = os.path.join(directory_path, '**', '*.jpg')

    # 使用glob.glob()查找所有匹配的文件，recursive=True允许搜索子目录
    jpg_files = glob.glob(pattern, recursive=True)

    return jpg_files

# 指定要遍历的文件夹路径
# directory_path = '/home/liuyang/datasets/silu__dataset_4k/'

# 调用函数并打印结果
img_files = collect_jpg_files_glob(directory_path)
img_files = [f for f in img_files if exclude not in f]
img_files = [Path(f).relative_to(relative_to) for f in img_files]

if Path(image_list).is_file():
    raise RuntimeError(f"{image_list} exist!")
else:
    with open(str(image_list), "w") as f:
        for img_p in img_files:
            f.write(f"./{str(img_p)}")
            f.write("\n")


random.shuffle(img_files)
img_num = len(img_files)
train_num = int(0.8 * img_num)

train_files = img_files[:train_num]
test_files = img_files[train_num:]

train_list = Path(image_list).with_name("train_list.txt")
test_list = Path(image_list).with_name("test_list.txt")

if Path(train_list).is_file():
    raise RuntimeError(f"{image_list} exist!")
else:
    with open(str(train_list), "w") as f:
        for img_p in train_files:
            f.write(f"./{str(img_p)}")
            f.write("\n")

if Path(test_list).is_file():
    raise RuntimeError(f"{image_list} exist!")
else:
    with open(str(test_list), "w") as f:
        for img_p in test_files:
            f.write(f"./{str(img_p)}")
            f.write("\n")
