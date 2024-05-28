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
label_set_all = set({})
for l in yolo_ds.labels:
    label_set = set(l['cls'].flatten())
    label_set_all = label_set_all.union(label_set)
import pdb; pdb.set_trace()
print(label_set_all)
print(len(label_set_all))

selected_ids = set([2, 5, 7])
selected_imgs = []
for l in yolo_ds.labels:
    cls = set(l['cls'].astype(int).flatten())
    if len(selected_ids.intersection(cls)):
        selected_imgs.append(l['im_file'])


def xywhn2xyxy(xywhn, h=640, w=640):
    xyxy = np.empty_like(xywhn)
    xyxy[..., 0] = w * (xywhn[..., 0] - xywhn[..., 2] / 2)
    xyxy[..., 1] = h * (xywhn[..., 1] - xywhn[..., 3] / 2)
    xyxy[..., 2] = w * (xywhn[..., 0] + xywhn[..., 2] / 2)
    xyxy[..., 3] = h * (xywhn[..., 1] + xywhn[..., 3] / 2)
    return xyxy.astype(int)

for l in yolo_ds.labels:
    cls = l['cls'].flatten()
    cls_set = set(cls)
    if not len(cls_set.intersection(l_uncompleted)):
        continue

    print(l['shape'])
    img = cv2.imread(l['im_file'])
    for i, c in enumerate(cls):
        if c in l_uncompleted:
            # get xywhn
            xywhn = l['bboxes'][i]
            x0, y0, x1, y1 = xywhn2xyxy(xywhn, h=l['shape'][0], w=l['shape'][1])
            roi = img[y0:y1, x0:x1]
            roi_images[c].append(roi)
            if len(roi_images[c]) >= 20:
                l_uncompleted.remove(c)
    if not len(l_uncompleted):
        break

result_dir = Path(f"./results/{split}")
result_dir.mkdir(parents=True, exist_ok=True)

for l, images in roi_images.items():
    for i, img in enumerate(images):
        cv2.imwrite(str(result_dir / f"label_{l}_img_{i}.jpg"), img)

