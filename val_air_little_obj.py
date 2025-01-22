from ultralytics import YOLO

data = "./datasets-config/air_little_obj.yaml"
model = YOLO('../yolo_models/yolov8n_sod4bird_finetune.pt')

args = {"data": data, "project": "yolov8-air_little_obj",
        "imgsz": 640, "name": "sod4bird_finetune", 'batch': 1}
results = model.val(**args)


