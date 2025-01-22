from ultralytics import YOLO

data = "./datasets-config/air_little_obj_roi-yanshou-10-percent.yaml"
model = YOLO('../yolo_models/yolov8n.pt')

args = {"data": data, "epochs": 300, "project": "yolov8-air_little_obj_roi",
        "imgsz": 32, "name": "yolov8n-32_yanshou_10-percent_", 'batch': 2500}
results = model.train(**args)


