from ultralytics import YOLO
from pathlib import Path

model = YOLO("/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj/sky/weights/best.pt")

args = {"project": "runs_yolov8-air_little_obj",
        "device": "0", "batch": 1, 'augment': False, 'imgsz': 640, 'conf': 0.2, 'name': 'pred-tmp',
        "save": True, "source": "/home/liuyang/Downloads/yanshou_yuxiaonan/tmp/frame_0001.jpg" }
results = model.predict(**args)


