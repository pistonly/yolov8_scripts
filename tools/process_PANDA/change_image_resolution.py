from pathlib import Path
import glob
import cv2
import json
import numpy as np

PANDA_dir = Path("/home/liuyang/datasets/PANDA/")

image_train = PANDA_dir / "image_train"
image_test = PANDA_dir / "image_test"
image_annos = PANDA_dir / "image_annos"

new_resolution = 1280

yolo_label_dir = PANDA_dir / "yolo_labels"
yolo_label_dir.mkdir(parents=True, exist_ok=True)

target_dir = PANDA_dir / f"images_{new_resolution}"
target_dir.mkdir(exist_ok=True, parents=True)

train_images = glob.glob(f"{str(image_train)}/**/*.jpg")

image_train_str = str(image_train)
for img_f in train_images:
    new_path = target_dir / img_f[len(image_train_str) + 1:]
    img_dir = new_path.parent
    img_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(img_f)
    h, w = img.shape[:2]
    rh = new_resolution / h
    rw = new_resolution / w
    r = min(rh, rw)
    h_new = int(h * r)
    w_new = int(w * r)
    img = cv2.resize(img, (w_new, h_new))
    cv2.imwrite(str(new_path), img)

