from ultralytics import YOLO
from pathlib import Path


model_dir = "/mnt/e/nfs_share/yolo-pt-models"
models = [f for f in Path(model_dir).iterdir() if str(f).endswith(".pt")]

source = "/home/qy/Downloads/20241110/20241110072211416-31-2-main"
for model_f in models:
    model = YOLO(str(model_f))
    args = {"source": source, "project": "yolov8_yanshou-20240922-0930-babiao",
            "imgsz": 1920, "name": f"20241110072211416-31-2-main_{model_f.stem}_", 'batch': 1, 'conf': 0.01, "save": True, 'line_width': 1}

    results = model.predict(**args)
