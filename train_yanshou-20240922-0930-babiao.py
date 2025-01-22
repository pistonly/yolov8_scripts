from ultralytics import YOLO

data = "./datasets-config/yanshou_20240922_0930_babiao_v2.yaml"
model = YOLO("./yolov8_yanshou-20240922-0930/yolov8n-continue-train-val-no-overlap/weights/best.pt")

args = {"data": data, "epochs": 300, "project": "yolov8_yanshou-20240922-0930-babiao",
        "imgsz": 2560, "name": "yolov8n-continue-train-val-no-overlap", 'batch': 2 * 9, "device": "0,1,2,3,4,5,6,7,8"}
results = model.train(**args)


