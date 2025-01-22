from ultralytics import YOLO

data = "./datasets-config/yanshou_laixi_20241122.yaml"
model = YOLO('../yolo_models/yolov8n.pt')

args = {"data": data, "epochs": 300, "project": "runs_yanshou_laixi_1122",
        "imgsz": 1920, "name": "batch-36_sz-1920_freeze-10_lr0-0005_", 'batch': 4 * 9, "device": "0,1,2,3,4,5,6,7,8",
        "freeze": 10, "optimizer": "SGD", "lr0": 0.005}
# args = {"data": data, "epochs": 300, "project": "runs_yanshou_laixi_1122",
#         "imgsz": 1920, "name": "test_", 'batch': 2, "device": "0"}
results = model.train(**args)


