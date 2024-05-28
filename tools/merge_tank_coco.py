from ultralytics.data import YOLODataset
import cv2
import numpy as np
from pathlib import Path

data = {"path": "/home/liuyang/datasets/coco",
        "train": "/home/liuyang/datasets/coco/COCO2017train/images",
        "val": "/home/liuyang/datasets/coco/COCO2017val/images",
        # "val": "/data/silu/dataset_4k/images/val",
        # "test": "/data/silu/dataset_4k/images/test",
        "names": range(80)
        }
split = "val"
yolo_ds = YOLODataset(data=data, img_path=data[split])

selected_ids = set([2, 5, 7])
selected_imgs = []
for l in yolo_ds.labels:
    cls = set(l['cls'].astype(int).flatten())
    if len(selected_ids.intersection(cls)):
        selected_imgs.append(l['im_file'])
selected_img_len = len(selected_imgs)


def xywhn2xyxy(xywhn, h=640, w=640):
    xyxy = np.empty_like(xywhn)
    xyxy[..., 0] = w * (xywhn[..., 0] - xywhn[..., 2] / 2)
    xyxy[..., 1] = h * (xywhn[..., 1] - xywhn[..., 3] / 2)
    xyxy[..., 2] = w * (xywhn[..., 0] + xywhn[..., 2] / 2)
    xyxy[..., 3] = h * (xywhn[..., 1] + xywhn[..., 3] / 2)
    return xyxy

def xyxy2xywhn(xyxy, h=640, w=640):
    xywhn = np.empty_like(xyxy)
    xywhn[..., 0] = (xyxy[..., 0] + xyxy[..., 2]) / 2 / w
    xywhn[..., 1] = (xyxy[..., 1] + xyxy[..., 3]) / 2 / h
    xywhn[..., 2] = (xyxy[..., 2] - xyxy[..., 0]) / w
    xywhn[..., 3] = (xyxy[..., 3] - xyxy[..., 1]) / h
    return xywhn

tk_data = {"path": "/home/liuyang/datasets/tank_dataset",
           "val": "/home/liuyang/datasets/tank_dataset/valid/images",
           # "val": "/data/silu/dataset_4k/images/val",
           # "test": "/data/silu/dataset_4k/images/test",
           "names": range(1)
           }

result_dir = Path("./tank_val_new")
result_img_dir = result_dir / "images"
result_lab_dir = result_dir / "labels"
result_img_dir.mkdir(parents=True, exist_ok=True)
result_lab_dir.mkdir(parents=True, exist_ok=True)

tank_ds = YOLODataset(data=data, img_path=tk_data['val'])
for i, l in enumerate(tank_ds.labels):
    im_file = l['im_file']
    img = cv2.imread(im_file)
    h, w = img.shape[0:2]
    car_img = cv2.imread(selected_imgs[i % selected_img_len])
    h1, w1 = car_img.shape[0:2]
    ratio = w * 1. / w1
    h_target = int(ratio * h1)
    w_target = w
    car_img_new = cv2.resize(car_img, (w_target, h_target))
    img_new = np.concatenate([img, car_img_new], axis=0)
    h_new, w_new = img_new.shape[:2]

    im_file = Path(im_file)
    cv2.imwrite(str(result_img_dir / im_file.name), img_new)
    print(im_file)

    # re label
    cls = l['cls'].flatten()
    with open(str(result_lab_dir / f"{im_file.stem}.txt"), "w") as label_f:
        for i, c in enumerate(cls):
            xywhn = l['bboxes'][i]
            xyxy = xywhn2xyxy(xywhn, h=l['shape'][0], w=l['shape'][1])
            xywhn_1 = xyxy2xywhn(xyxy, h=h_new, w=w_new)
            xywhn_1 = xywhn_1.flatten()
            label_f.write(f"{int(c)} {xywhn_1[0]:.4f} {xywhn_1[1]:.4f} {xywhn_1[2]:.4f} {xywhn_1[3]:.4f}\n")


