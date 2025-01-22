from ultralytics import YOLO

data = "./datasets-config/sky.yaml"
model = YOLO('../yolo_models/yolov8n.pt')

args = {"data": data, "epochs": 300, "project": "runs_yolov8-air_little_obj",
        "imgsz": 640, "name": "sky", 'batch': 16}
results = model.train(**args)


