from ultralytics import YOLO

data = "./datasets-config/silu_dataset_4k__with_hms_finetune.yaml"
model = YOLO('../yolo_models/yolov8n-silu-4k-hms.pt')

args = {"data": data, "epochs": 10, "project": "yolov8-silu-4k-hms",
        "imgsz": 1280, "name": "yolov8n-finetune", 'batch': 4, "lr0": 0.0001}
results = model.train(**args)


