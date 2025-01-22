from ultralytics import YOLO

data = "./datasets-config/yanshou_20240922_split-1280.yaml"
model = YOLO('../yolo_models/yolov8n.pt')

args = {"data": data, "epochs": 300, "project": "20240922_split-1280",
        "imgsz": 1280, "name": "yolov8n", 'batch': 10 * 9, "device": "0,1,2,3,4,5,6,7,8"}
results = model.train(**args)


