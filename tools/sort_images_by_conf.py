import numpy as np
from pathlib import Path
import pandas as pd

yolo_predict_dir = Path("/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/predict_als-1216_sz-32_cls-2_exclude-small_overfit_04/labels/")

def read_one_txt(txt_file: Path):
    yolo_res = np.loadtxt(str(txt_file))
    yolo_res = yolo_res.reshape((-1, 6))
    ind = np.argmax(yolo_res[:, -1])
    _id = yolo_res[ind, 0]
    conf = yolo_res[ind, -1]
    return conf, _id

conf_data = []
for txt_file in yolo_predict_dir.iterdir():
    conf_data.append([txt_file.with_suffix(".jpg").name, *read_one_txt(txt_file)])

conf_data = sorted(conf_data, key=lambda x: x[1], reverse=True)

conf_data = pd.DataFrame(conf_data, columns=["stem", "conf", "id"])
conf_data.to_csv("./results/sorted_images.txt", index=False)
