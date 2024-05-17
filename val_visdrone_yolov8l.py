from ultralytics import YOLO


model = YOLO('/mnt/hd/Documents/YOLO/yolov8_scripts/yolov8-visdrone/yolov8l/weights/best.pt')

args = {"data": "VisDrone.yaml", "project": "yolov8-visdrone",
        "device": "0", "batch": 1, 'augment': False, 'imgsz': 2016, 'name': 'val-yolov8l-640train-2016t'}
results = model.val(**args)


