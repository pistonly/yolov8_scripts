from ultralytics.data import YOLODataset
import cv2
import numpy as np
from pathlib import Path
import yaml

data = "../datasets-config/panda_one_label_gega.yaml"
split = "train"

with open(data) as d_f:
    data_dict = yaml.safe_load(d_f)

dataset_name = Path(data).name
min_size = 0
output_dir = Path(f"./results/{dataset_name}")
output_dir.mkdir(parents=True, exist_ok=True)

data_args = {"img_path": str(Path(data_dict['path']) / data_dict[split]),
             "data": data_dict}
yolo_ds = YOLODataset(**data_args)
label_set_all = set({})
for l in yolo_ds.labels:
    label_set = set(l['cls'].flatten())
    label_set_all = label_set_all.union(label_set)
print(label_set_all)
print(len(label_set_all))

# cut roi
roi_images = {}
for l in label_set_all:
    roi_images.update({l: []})
l_uncompleted = label_set_all.copy()

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
            _min_size = min(y1 - y0, x1 - x0)
            if _min_size > min_size:
                roi_images[c].append([])
                cv2.imwrite(str(output_dir / f"label_{c}_img_{len(roi_images[c])}.jpg"), roi)
            # if len(roi_images[c]) >= 20:
            #     l_uncompleted.remove(c)
    if not len(l_uncompleted):
        break

# for l, images in roi_images.items():
#     for i, img in enumerate(images):
#         cv2.imwrite(str(output_dir / f"label_{l}_img_{i}.jpg"), img)

