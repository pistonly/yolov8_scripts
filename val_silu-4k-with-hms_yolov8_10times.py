from ultralytics import YOLO

for i in range(10, 15):
    data = f"./datasets-config/hms_{i}.yaml"
    model = YOLO('./yolov8-silu-4k-hms/yolov8n-finetune3/weights/last.pt')
    # model = YOLO('../yolo_models/yolov8n-silu-4k-hms.pt')

    args = {"data": data, "project": "yolov8-silu-4k-hms",
            "imgsz": 1280, "name": "val_yolov8n-finetune", 'batch': 4}
    results = model.val(**args)


