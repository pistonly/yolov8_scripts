from ultralytics import YOLO

data = "./datasets-config/air_little_obj_roi_als.yaml"
model = YOLO('../yolo_models/yolov8n.pt')

# args = {"data": data, "epochs": 600, "project": "runs_yolov8-air_little_obj_roi",
#         "imgsz": 32, "name": "als-1214_sz-32_freeze-10_lr0-0003_", 'batch': 2048,
#         "freeze": 10, "optimizer": "SGD", "lr0": 0.003}
# args = {"data": data, "epochs": 300, "project": "runs_yolov8-air_little_obj_roi",
#         "imgsz": 32, "name": "als-1214_sz-32_lr0-001_", 'batch': 2048,
#         "optimizer": "SGD", "lr0": 0.01}
args = {"data": data, "epochs": 300, "project": "runs_yolov8-air_little_obj_roi",
        "imgsz": 32, "name": "als-1214_sz-32_lr0-001-add_some_images_", 'batch': 2048,
        "optimizer": "SGD", "lr0": 0.01}
results = model.train(**args)


