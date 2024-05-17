import cv2
import numpy as np
from pathlib import Path
import re


def plot_images(imgs, grid_size=(4, 5), grid_shape=(100, 100)):

    def resize_tile(img):
        h, w = img.shape[0:2]
        r_h = grid_shape[0] / h
        r_w = grid_shape[1] / w
        r_min = min(r_h, r_w)
        if r_min < 1:
            h_new = int(h * r_min)
            w_new = int(w * r_min)
            img = cv2.resize(img, (w_new, h_new))
        return img

    imgH = grid_size[0] * grid_shape[0]
    imgW = grid_size[1] * grid_shape[1]
    image = np.zeros((imgH, imgW, 3), np.uint8)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            grid_center_y = grid_shape[0] // 2 + i * grid_shape[0]
            grid_center_x = grid_shape[1] // 2 + j * grid_shape[1]
            img_id = i * grid_size[1] + j
            if img_id >= len(imgs):
                break
            img_i = resize_tile(imgs[img_id])
            h, w = img_i.shape[0:2]
            upleft_y = grid_center_y - h // 2
            upleft_x = grid_center_x - w // 2
            image[upleft_y:upleft_y + h, upleft_x:upleft_x + w] = img_i
    return image

img_dir = Path("/home/liuyang/Documents/YOLO/yolov8_scripts/tools/results/")

re_pattern = r"label_([0-9\.]+)_img_([0-9]+).jpg"
img_all = {}
for img_path in img_dir.iterdir():
    img_name = img_path.name
    res = re.match(re_pattern, img_name)
    if res:
        cls, img_id = int(float(res.groups()[0])), int(res.groups()[1])
        img = cv2.imread(str(img_path))
        if cls not in img_all:
            img_all[cls] = []
        img_all[cls].append((img_id, img))

for k, v in img_all.items():
    v.sort(key=lambda x: x[0])
    imgs = [v_i[1] for v_i in v]
    image = plot_images(imgs, (4, 5))
    cv2.imwrite(str(img_dir / f"merged_l_{k:d}.jpg"), image)
