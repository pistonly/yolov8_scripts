from ultralytics import YOLO

data = "./datasets-config/hsh_second.yaml"
model = YOLO('../yolo_models/yolov8m.pt')

args = {"data": data, "epochs": 300, "project": "yolov8-hsh-second",
        "imgsz": 1280, "name": "yolov8m", 'batch': 8}
results = model.train(**args)


