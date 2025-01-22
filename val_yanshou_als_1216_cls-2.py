from ultralytics import YOLO

import ultralytics
print(ultralytics.__file__)

data = "/home/liuyang/Documents/YOLO/yolov8_scripts/datasets-config/air_little_obj_als_tmp.yaml"
# data = "/home/liuyang/Documents/YOLO/yolov8_scripts/datasets-config/air_little_obj_roi_als_cls-2_exclude-small_overfit.yaml"
model = YOLO('/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1216_sz-32_cls-2_exclude-small_overfit_0/weights/best.pt')

args = {"data": data, "project": "runs_yolov8-air_little_obj_roi",
        "imgsz": 32, "name": "val_als-1216_sz-32_cls-2_exclude-small_overfit_0", 'batch': 9}
results = model.val(**args)


