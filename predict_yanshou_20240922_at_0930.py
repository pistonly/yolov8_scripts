from ultralytics import YOLO

model = YOLO('../yolo_models/yolov8n_yanshou_20240922_best_2560.pt')

source = "/home/liuyang/datasets/20240930/images/"
args = {"source": source, "project": "yolov8_yanshou-20240922",
        "imgsz": 2560, "name": "predict_yolov8n-2560-20240922_at_20240930", 'batch': 1,
        "save_frames": True, "save": True
        }
results = model.predict(**args)


