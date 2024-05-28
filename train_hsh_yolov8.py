from ultralytics import YOLO

data = "./datasets-config/hsh_first.yaml"
model = YOLO('../yolo_models/yolov8n.pt')

args = {"data": data, "epochs": 300, "project": "yolov8-hsh",
        "imgsz": 1280, "name": "yolov8n", 'batch': 16}
results = model.train(**args)


