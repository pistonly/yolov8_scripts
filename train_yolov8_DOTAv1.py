from ultralytics import YOLO

data = "./datasets-config/my_DOTAv1.yaml"
model = YOLO('../yolo_models/yolov8n.pt')

args = {"data": data, "epochs": 300, "project": "yolov8_DOTAv1",
        "imgsz": 1024, "name": "yolov8n-1024_", 'batch': 4 * 9, "device": "0,1,2,3,4,5,6,7,8"}
results = model.train(**args)


