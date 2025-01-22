from ultralytics import YOLO

data = "./datasets-config/yanshou_20240922_0930_v2_vehicle.yaml"
model = YOLO('../yolo_models/yolov8n.pt')

args = {"data": data, "epochs": 300, "project": "yanshou_vehicle",
        "imgsz": 640, "name": "yolov8n-20240922-0930-vehicle-640_", 'batch': 16, "device": "0"}
results = model.train(**args)


