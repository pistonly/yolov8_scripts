from ultralytics import YOLO
from pathlib import Path

current_dir = Path(__file__).absolute().parent
model_dir = current_dir.parent / "yolo_models"

model = YOLO(str(model_dir / "yolov8l.pt"))
# model = YOLO("/home/liuyang/Documents/YOLO/yolov8_scripts/yolov8-silu-4k-hms/yolov8l7/weights/best.pt")

args = {"project": "results/tmp",
        "device": "0", "batch": 1, 'augment': False, 'imgsz': 640, 'conf': 0.6, 'name': 'pred-yolov8l-640-bird',
        "save": True, "source": "/media/liuyang/WD_BLACK/liuyang/datasets/drone-bird/Dataset/valid/images/", "save_txt": True}
results = model.predict(**args)


