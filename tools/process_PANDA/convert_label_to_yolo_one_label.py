from pathlib import Path
import glob
import cv2
import json
import numpy as np

PANDA_dir = Path("/home/liuyang/datasets/PANDA/")

image_train = PANDA_dir / "image_train"
image_test = PANDA_dir / "image_test"
image_annos = PANDA_dir / "image_annos"


test_images = glob.glob(f"{str(image_test)}/**/*.jpg")
img = cv2.imread(test_images[0])
h, w = img.shape[0:2]
print(f"h: {h}, w: {w}")


# label_name = "person_bbox_train"
label_name = "vehicle_bbox_train"
train_label_file = f"/home/liuyang/datasets/PANDA/image_annos/{label_name}.json"
train_labels = json.load(open(train_label_file))

yolo_label_dir = PANDA_dir / "yolo_labels" / f"{label_name}_one_label_new" 
yolo_label_dir.mkdir(parents=True, exist_ok=True)

category_all = set()
label_yolo = {}

def xyxyn2xywhn(xyxyn):
    xywhn = np.empty_like(xyxyn)
    xywhn[..., 0] = (xyxyn[..., 0] + xyxyn[..., 2]) / 2 
    xywhn[..., 1] = (xyxyn[..., 1] + xyxyn[..., 3]) / 2
    xywhn[..., 2] = xyxyn[..., 2] - xyxyn[..., 0]
    xywhn[..., 3] = xyxyn[..., 3] - xyxyn[..., 1]
    xywhn = xywhn.clip(0, 1)
    if min(xywhn) < 0:
        print(min(xywhn))
    return xywhn

def get_xyxyn(obj: dict):
    if 'rect' in obj:
        x0 = obj['rect']['tl']['x']
        y0 = obj['rect']['tl']['y']
        x1 = obj['rect']['br']['x']
        y1 = obj['rect']['br']['y']
        return np.array([x0, y0, x1, y1])
    else:
        raise RuntimeError(f"rect not found in obj: {obj}")


for img_f, label in train_labels.items():
    if 'objects list' in label:
        obj_list = label["objects list"]
        xywhn_list = []
        category_list = []
        for obj in obj_list:
            category = obj['category']
            xyxyn = get_xyxyn(obj)
            xywhn = xyxyn2xywhn(xyxyn)
            xywhn_list.append(xywhn)
            category_list.append(category)
            category_all.add(category)
        label_yolo[img_f] = {"cls": category_list, "bbox": xywhn_list}

category_map = dict(zip(category_all, np.zeros(len(category_all), dtype=int)))

for img_f, label in label_yolo.items():
    label_file = (yolo_label_dir / img_f).with_suffix(".txt")
    label_dir = label_file.parent
    label_dir.mkdir(parents=True, exist_ok=True)
    with open(str(label_file), "w") as f:
        for cls_i, bbox_i in zip(label['cls'], label['bbox']):
            f.write(f"{category_map[cls_i]} {bbox_i[0]} {bbox_i[1]} {bbox_i[2]} {bbox_i[3]}\n")

print(category_map)
with open(str(yolo_label_dir / f"{label_name}_names.txt"), "w") as f:
    for c, _id in category_map.items():
        f.write(f"{_id}: {c}\n")
