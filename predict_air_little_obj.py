from ultralytics import YOLO

# model = YOLO("/home/liuyang/Documents/YOLO/yolov8_scripts/yolov8-air_little_obj_roi/yolov8n7/weights/best.pt")
model = YOLO("/home/liuyang/Documents/YOLO/yolov8_scripts/yolov8-air_little_obj_roi/yolov8n-32_/weights/best.pt")
# model = YOLO('../yolo_models/yolov8n_sod4bird_finetune.pt')

source = "/home/liuyang/Documents/background_subtraction/python/merge_dataset_tmp/images/"
# source = "/home/liuyang/datasets/air_little_obj/images/"
args = {"source": source, "project": "yolov8-air_little_obj_roi",
        "imgsz": 640, "name": "pred_yolov8n_roi_to_merge_conf-0.01_", 'batch': 1, 'conf': 0.01, "save": True, 'line_width': 1}
results = model.predict(**args)


