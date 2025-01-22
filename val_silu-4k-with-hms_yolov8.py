from ultralytics import YOLO

data = "./datasets-config/silu_dataset_4k__with_hms_finetune.yaml"
# model = YOLO('./yolov8-silu-4k-hms/yolov8n-finetune3/weights/last.pt')
model = YOLO('../yolo_models/yolov8n-silu-4k-hms.pt')

args = {"data": data, "project": "yolov8-silu-4k-hms",
        "imgsz": 1280, "name": "val_yolov8n-finetune", 'batch': 4}
results = model.val(**args)


