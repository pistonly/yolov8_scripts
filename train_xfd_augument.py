from ultralytics import YOLO

data = "./datasets-config/xfd_augument.yaml"
model = YOLO('../yolo_models/yolov8n.pt')

args = {"data": data, "epochs": 300, "project": "runs-yolov8-air_little_obj_xfd_augument",
        "imgsz": 1280, "name": "yolov8n-1280-", 'batch': 10}
results = model.train(**args)


