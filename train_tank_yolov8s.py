from ultralytics import YOLO
from pathlib import Path

current_dir = Path(__file__).absolute().parent
model_dir = current_dir.parent / "yolo_models"

model = YOLO(str(model_dir / "yolov8l.pt"))

args = {"data": "./datasets-config/tank_dataset.yaml", "epochs": 200, "project": "yolov8_tank_dataset",
        "device": "0,1,2", "batch": 60 * 3, "name": "yolov8s", "imgsz": 640}
results = model.train(**args)


