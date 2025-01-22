from ultralytics.data import YOLODataset
import cv2
import numpy as np
from pathlib import Path
import yaml

# data = "../datasets-config/panda_one_label_gega.yaml"
data = "../datasets-config/VisDrone.yaml"
split = "train"

with open(data) as d_f:
    data_dict = yaml.safe_load(d_f)

dataset_name = Path(data).stem
min_size = 0
number_per_label = np.inf
canvas_size = None  # or None means "no canvas"
output_dir = Path(f"./results/{dataset_name}_{canvas_size}")
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

def get_canvas(roi, img_shape, canvas_size=1024, img=np.empty((640, 640))):
    canvas_size_bac = canvas_size
    x0, y0, x1, y1 = roi
    h_roi, w_roi = y1 - y0, x1 - x0
    if h_roi > canvas_size or w_roi > canvas_size:
        canvas_size = max(h_roi, w_roi)
    x_center = (x0 + x1) // 2
    y_center = (y0 + y1) // 2
    x0_canvas = np.clip(x_center - canvas_size // 2, 0, x0)
    y0_canvas = np.clip(y_center - canvas_size // 2, 0, y0)
    x1_canvas = x0_canvas + canvas_size
    y1_canvas = y0_canvas + canvas_size

    h, w = img_shape[:2]
    if x1_canvas > w:
        x1_canvas = w
        x0_canvas = w - canvas_size
    if y1_canvas > h:
        y1_canvas = h
        y0_canvas = h - canvas_size
    canvas = img[y0_canvas: y1_canvas, x0_canvas: x1_canvas]
    if canvas_size > canvas_size_bac:
        canvas = cv2.resize(canvas, (canvas_size_bac, canvas_size_bac))
    return canvas

for l in yolo_ds.labels:
    cls = l['cls'].flatten()
    cls_set = set(cls)
    if not len(cls_set.intersection(l_uncompleted)):
        continue

    print(l['shape'])
    img = cv2.imread(l['im_file'])
    img_name = Path(l['im_file']).stem
    h, w = img.shape[0:2]
    for i, c in enumerate(cls):
        if c in l_uncompleted:
            # get xywhn
            xywhn = l['bboxes'][i]
            x0, y0, x1, y1 = xywhn2xyxy(xywhn, h=h, w=w)
            _min_size = min(y1 - y0, x1 - x0)
            if _min_size > min_size:
                if canvas_size:
                    roi = get_canvas((x0, y0, x1, y1), (h, w), canvas_size, img)
                else:
                    roi = img[y0:y1, x0:x1]
                roi_images[c].append([])
                cv2.imwrite(str(output_dir / f"label_{int(c)}_img_{img_name}-{i}.jpg"), roi)
            if len(roi_images[c]) >= number_per_label:
                l_uncompleted.remove(c)
    if not len(l_uncompleted):
        break


