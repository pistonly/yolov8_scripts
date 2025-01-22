from ultralytics import YOLO

data = "./datasets-config/air_little_obj_roi.yaml"
model = YOLO('../yolo_models/yolov8n.pt')
# model = YOLO('/home/liuyang/Documents/YOLO/yolov8_scripts/yolov8-air_little_obj/yolov8n3/weights/best.pt')

args = {"data": data, "epochs": 300, "project": "yolov8-air_little_obj_roi",
        "imgsz": 32, "name": "yolov8n-32_", 'batch': 2515}
results = model.train(**args)


