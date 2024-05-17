from ultralytics import YOLOWorld

data = "./datasets-config/VisDrone.yaml"

model = YOLOWorld('../yolo_models/yolov8s-worldv2.pt')
# args = dict(data=data, epochs=100, imgsz=1280, optimizer='SGD', lr0=0.001, lrf=0.05)
args = dict(data=data, epochs=100, imgsz=1280, batch=8 * 3, device="2,4,5", project="yolo-world_fine-tune_VisDrone", name="yolov8s-worldv2_")
results = model.train(**args)
