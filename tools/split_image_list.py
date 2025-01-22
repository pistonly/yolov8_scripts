import random
import numpy as np
from pathlib import Path

image_list_path = "/home/liuyang/Documents/SOT/8-visualize_track_results/python/yolo_ds/image_list.txt"
image_list = np.loadtxt(image_list_path, str)

random.shuffle(image_list)
image_num = len(image_list)
train_num = int(image_num * 0.8)
train_list = image_list[0:train_num]
val_list = image_list[train_num:]

train_list_path = Path(image_list_path).parent / "train_images.txt"
val_list_path = Path(image_list_path).parent / "val_images.txt"

if train_list_path.is_file():
    raise RuntimeError(f"{train_list_path} exist")
else:
    with open(str(train_list_path), "w") as f:
        for img_p in train_list:
            f.write(img_p)
            f.write("\n")

if val_list_path.is_file():
    raise RuntimeError(f"{val_list_path} exist")
else:
    with open(str(val_list_path), "w") as f:
        for img_p in val_list:
            f.write(img_p)
            f.write("\n")
