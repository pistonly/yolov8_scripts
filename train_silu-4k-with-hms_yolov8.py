from ultralytics import YOLO

data = "./datasets-config/silu_dataset_4k__with_hms.yaml"
model = YOLO('../yolo_models/yolov8n.pt')

args = {"data": data, "epochs": 300, "project": "yolov8-silu-4k-hms",
        "imgsz": 1280, "name": "yolov8n", 'batch': 4}
results = model.train(**args)


