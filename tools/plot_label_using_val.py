from pathlib import Path
from ultralytics import YOLOWorld


current_dir = Path(__file__).absolute().parent
model_dir = current_dir.parent.parent / "yolo_models"

model = YOLOWorld(str(model_dir / "yolov8s-worldv2.pt"))
# model.set_classes(["tk"])
model.set_classes(["person", "car", "truck", "boat", "airplane", "train", "bicycle", "tricycle"])
args = {"data": "./data_plot_images/plot_data.yaml", "device": "0", "project": "plot_dataset",
        "name": "tmp", "batch": 1, "imgsz": 1280, "split": "val", }

metrics = model.val(**args)


