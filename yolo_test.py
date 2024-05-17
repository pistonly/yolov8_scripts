from ultralytics import YOLO

data = "coco8.yaml"

model = YOLO('../yolo_models/yolov8n.pt')
# args = dict(data=data, epochs=100, imgsz=1280, optimizer='SGD', lr0=0.001, lrf=0.05)
args = dict(data=data, epochs=1, project="run-test")
results = model.train(**args)
