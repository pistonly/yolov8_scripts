from ultralytics import YOLO

# model = YOLO('./yolov8_yanshou-20240922-0930/yolov8n-continue-train-val-no-overlap/weights/best.pt')
model = YOLO("/home/liuyang/Documents/qiyuan_jiaojie/nnn_om_convert/models/yolov8n_best_kxw.pt")

source = "/home/liuyang/Documents/qiyuan_jiaojie/SOT/data/20241017180027130-11-1-main-selected/"
args = {"source": source, "project": "yolov8_yanshou-20240922",
        "imgsz": 2560, "name": "predict_yolov8n-best-kxw_at_sot-0", 'batch': 1,
        "save_frames": True, "save": True, "conf": 0.01
        }
results = model.predict(**args)


