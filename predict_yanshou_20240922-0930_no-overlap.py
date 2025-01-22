from ultralytics import YOLO

model = YOLO('./yolov8_yanshou-20240922-0930/yolov8n-continue-train-val-no-overlap/weights/best.pt')

source = "/home/liuyang/datasets/20240930/images/"
args = {"source": source, "project": "yolov8_yanshou-20240922",
        "imgsz": 2560, "name": "predict_yolov8n-2560-merge-no-overlap_at_20240930", 'batch': 1,
        "save_frames": True, "save": True
        }
results = model.predict(**args)


