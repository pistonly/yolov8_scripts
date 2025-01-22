from ultralytics import YOLO

data = "./datasets-config/yanshou_20240930.yaml"
model = YOLO('./yolov8_yanshou-20240922-0930/yolov8n-continue2/weights/best.pt')

args = {"data": data, "project": "yolov8_yanshou-20240922",
        "imgsz": 2560, "name": "val_yolov8n-2560_at_20240930_merge_train", 'batch': 1}
results = model.val(**args)


