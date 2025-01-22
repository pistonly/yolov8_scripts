from ultralytics import YOLO

data = "./datasets-config/my_DOTAv1.yaml"
model = YOLO('../yolo_models/yolo11n.pt')

args = {"data": data, "epochs": 300, "project": "yolo11_DOTAv1",
        "imgsz": 1024, "name": "yolo11n-1024_", 'batch': 4 * 9, "device": "0,1,2,3,4,5,6,7,8"}
results = model.train(**args)


