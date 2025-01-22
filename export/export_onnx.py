from ultralytics import YOLO


# model = YOLO("../../yolo_models/hsh_18_model.pt")
# model = YOLO("../../yolo_models/yolov8n_air-little-obj_32-roi.pt")
# model = YOLO("/home/liuyang/Documents/YOLO/yolov8_scripts/yolov8-air_little_obj_roi/yolov8n-32_/weights/best.pt")
# model = YOLO("/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1214_sz-32_lr0-0005_/weights/best.pt")
# model = YOLO("/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1214_sz-32_lr0-001_/weights/best.pt")
# model = YOLO("/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1216_sz-32_lr0-001-add_some_images_/weights/best.pt")
model = YOLO("/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj/sky/weights/best.pt")

# success = model.export(format="onnx", imgsz=(1088, 1920), opset=12)
success = model.export(format="onnx", imgsz=(640, 640), opset=12)
