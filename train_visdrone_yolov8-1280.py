from ultralytics import YOLO


model = YOLO('yolov8l.pt')

args = {"data": "VisDrone.yaml", "epochs": 150, "project": "yolov8-visdrone",
        "device": "0,1,2,3,4,5,6,7,8,9", "batch": 4 * 10, "name": "yolov8l-1280", "imgsz": 1280}
results = model.train(**args)


