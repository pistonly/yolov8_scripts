from ultralytics import YOLO


model = YOLO('yolov8l.pt')

args = {"data": "VisDrone.yaml", "epochs": 200, "project": "yolov8-visdrone",
        "device": "0,1,2,3,4,5,6,7,8,9", "batch": 16 * 10, "name": "yolov8l"}
results = model.train(**args)


