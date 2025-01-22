from ultralytics import YOLO

model = YOLO('./yolov8-silu-4k-hms/yolov8n-finetune3/weights/last.pt')

args = {"source": "/home/liuyang/datasets/haimasi_bac/full_images/", "project": "yolov8-silu-4k-hms",
        "imgsz": 1280, "name": "predict_yanshou_yolov8n-finetune", 'batch': 4, "save": True, "save_txt": True, "conf": 0.25}

results = model.predict(**args)


