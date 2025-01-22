from ultralytics import YOLO

data = "./datasets-config/silu_dataset_4k__with_hms.yaml"
model = YOLO('../yolo_models/yolov8l.pt')

args = {"data": data, "epochs": 300, "project": "yolov8-silu-4k-hms",
        "imgsz": 1280, "name": "yolov8l", 'batch': 3 * 3, "device": "0,2,3"}
results = model.train(**args)


