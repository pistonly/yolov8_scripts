from ultralytics import YOLO


model = YOLO('yolov8n.pt')

args = {"data": "VisDrone.yaml", "epochs": 300, "project": "yolov8-visdrone",
        "device": "6,7,8,9", "batch": 16 * 4}
results = model.train(**args)


