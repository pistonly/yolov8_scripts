from ultralytics import YOLO
from pathlib import Path

current_dir = Path(__file__).absolute().parent
model_dir = current_dir.parent / "yolo_models"

model = YOLO(str(model_dir / "yolov8l-tk.pt"))

args = {"project": "results/yolov8_tk_dataset",
        "device": "0", "batch": 1, 'augment': False, 'imgsz': 640, 'name': 'pred-tk_l-640train-merge_val_silu4k-640-',
        "split": "test", "save": True, "source": "/home/liuyang/datasets/tk_dataset/merge_valid_silu4k/images"}
results = model.predict(**args)


