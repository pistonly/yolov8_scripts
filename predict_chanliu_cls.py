from ultralytics import YOLO

model = YOLO("/home/liuyang/Documents/YOLO/yolov8_scripts/chanliu_tianshui/yolo11n-cls-b64-e300_2/weights/best.pt")  # load a pretrained model (recommended for training)

data = "/home/liuyang/datasets/quantaeye/chanliu_spaceA-18_cls/splitted_cls_dataset/train"

args = {"source": data, "project": "chanliu_tianshui",
        "imgsz": 640, "name": "predict_yolo11n-cls-b64_", 'batch': 1, "save": True}

results = model.predict(**args)
