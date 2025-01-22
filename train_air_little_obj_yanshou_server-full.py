from ultralytics import YOLO

data = "./datasets-config/xfd_augument.yaml"
model = YOLO('../yolo_models/yolov8n.pt')

args = {"data": data, "epochs": 300, "project": "yolov8-air_little_obj_roi",
        "imgsz": 32, "name": "yolov8n-32_yanshou_full_", 'batch': 2500, "patience": 300}
results = model.train(**args)


