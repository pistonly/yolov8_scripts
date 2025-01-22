from ultralytics import YOLO

data = "./datasets-config/yanshou_20240922.yaml"
model = YOLO('../yolo_models/yolov8n.pt')

args = {"data": data, "epochs": 300, "project": "yolov8_yanshou-20240922",
        "imgsz": 2560, "name": "yolov8n-2560_full-label", 'batch': 2 * 9, "device": "0,1,2,3,4,5,6,7,8"}
results = model.train(**args)


