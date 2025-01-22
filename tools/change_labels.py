import yaml
from pathlib import Path
import numpy as np
import cv2

label_map = {1: 0, 2: 1}

dataset_yaml = "../datasets-config/yanshou_20240922_0930_v2.yaml"
with open(dataset_yaml, encoding="utf-8") as f:
    s = f.read()
    data = yaml.safe_load(s)



def change_one_image(imagePath: Path, target_dir_name: str):
    if not imagePath.is_file():
        print(f"{str(imagePath)} is not file")
        return

    labelPath = imagePath.with_suffix(".txt")
    labelPath = str(labelPath).replace("images/", "labels/")
    target_dir = Path(labelPath).parents[1] / target_dir_name
    target_dir.mkdir(exist_ok=True)
    if Path(labelPath).is_file():
        labels_new = []
        with open(labelPath):
            labels = np.loadtxt(labelPath)
            labels = labels.reshape((1, -1)) if labels.ndim == 1 else labels
            for l in labels:
                if int(l[0]) in label_map:
                    l[0] = label_map[int(l[0])]
                    labels_new.append(l)

        labelPath = Path(labelPath)
        new_label_f = target_dir / labelPath.name
        with open(str(new_label_f), "w") as f:
            if len(labels_new):
                for l in labels_new:
                    f.write(f"{int(l[0])} {l[1]} {l[2]} {l[3]} {l[4]}\n")


image_list_file = Path(data['path']) / data['val']

if image_list_file.is_file():
    image_list = np.loadtxt(str(image_list_file), dtype=str)
else:
    image_list = []

for image_f in image_list:
    print(image_f)
    change_one_image(Path(data['path']) / image_f, "labels_vehicle")
