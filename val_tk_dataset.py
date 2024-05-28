from ultralytics import YOLO
from pathlib import Path

current_dir = Path(__file__).absolute().parent
model_dir = current_dir.parent / "yolo_models"

model = YOLO(str(model_dir / "yolov8l-tk.pt"))

args = {"data": "./datasets-config/tk_dataset.yaml", "project": "results/yolov8_tk_dataset",
        "device": "0", "batch": 1, 'augment': False, 'imgsz': 640, 'name': 'val-tank_l-640train-val640-',
        "split": "val"}
results = model.val(**args)


