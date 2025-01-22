from ultralytics import YOLO

data = "./datasets-config/air_little_obj_roi-yanshou-500m.yaml"
model = YOLO('../yolo_models/yolov8n.pt')
# model = YOLO('/home/liuyang/Documents/YOLO/yolov8_scripts/yolov8-air_little_obj/yolov8n3/weights/best.pt')

args = {"data": data, "epochs": 300, "project": "yolov8-air_little_obj_roi",
        "imgsz": 32, "name": "yolov8n-32_yanshou_", 'batch': 2500 * 9, "device": "0,1,2,3,4,5,6,7,8"}
results = model.train(**args)


