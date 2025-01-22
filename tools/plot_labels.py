import yaml
from pathlib import Path
import numpy as np
import cv2

dataset_yaml = "../datasets-config/yanshou_20240922.yaml"
with open(dataset_yaml, encoding="utf-8") as f:
    s = f.read()
    data = yaml.safe_load(s)

image_list_file = Path(data['path']) / data['train']

if image_list_file.is_file():
    image_list = np.loadtxt(str(image_list_file), dtype=str)
else:
    image_list = []

def xywhn2xyxy(xywhn: np.ndarray, H, W):
    xywhn[:, [0, 2]] *= W
    xywhn[:, [1, 3]] *= H
    xyxy = np.zeros_like(xywhn)
    xyxy[:, 0] = xywhn[:, 0] - xywhn[:, 2] / 2
    xyxy[:, 1] = xywhn[:, 1] - xywhn[:, 3] / 2
    xyxy[:, 2] = xywhn[:, 0] + xywhn[:, 2] / 2
    xyxy[:, 3] = xywhn[:, 1] + xywhn[:, 3] / 2
    return xyxy


def draw_one_image(imagePath: Path, target_dir: Path):
    if not imagePath.is_file():
        print(f"{str(imagePath)} is not file")
        return

    img = cv2.imread(str(imagePath))
    if img is None:
        print(f"read image: {str(imagePath)} error")
        return
    h, w = img.shape[:2]

    labelPath = imagePath.with_suffix(".txt")
    labelPath = str(labelPath).replace("images/", "labels/")
    if Path(labelPath).is_file():
        with open(labelPath):
            labels = np.loadtxt(labelPath)
            labels = labels.reshape((1, -1)) if labels.ndim == 1 else labels
    else:
        print(f"label file: {str(labelPath)} do not exist")
        labels = np.array([])

    target_dir.mkdir(exist_ok=True, parents=True)
    target_img_path = str(target_dir / imagePath.name)
    if len(labels) == 0:
        cv2.imwrite(target_img_path, img)
        return

    cls = labels[:, 0].astype(int)
    xywhn = labels[:, -4:].astype(float)
    xyxy = xywhn2xyxy(xywhn, h, w)
    xyxy = xyxy.astype(int)
    for l, xyxy_i in zip(cls, xyxy):
        cv2.rectangle(img, (xyxy_i[0], xyxy_i[1]), (xyxy_i[2], xyxy_i[3]), (255, 0, 0), 2)
        text = f"{data['names'][l]}"
        cv2.putText(img, text, (xyxy_i[0], xyxy_i[1] - 5), 0, 1, (255, 0, 0))

    cv2.imwrite(target_img_path, img)



target_dir = Path("./20240922_visual_tmp")
for image_f in image_list:
    print(image_f)
    draw_one_image(Path(data['path']) / image_f, target_dir)
