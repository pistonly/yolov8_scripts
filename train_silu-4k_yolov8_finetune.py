from ultralytics import YOLO

data = "./datasets-config/silu__dataset-4k_3.yaml"
model = YOLO('../yolo_models/yolov8n-silu-4k_epoch100.pt')

args = {"data": data, "epochs": 300, "project": "yolov8-silu-4k",
        "imgsz": 2048, "name": "yolov8n-finetune_imgsz_2048", 'batch': 1, "lr0": 0.01}
results = model.train(**args)


