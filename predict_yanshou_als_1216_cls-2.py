from ultralytics import YOLO

data = "./datasets-config/air_little_obj_roi_als_cls-2_exclude-small_overfit.yaml"
model = YOLO('/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yolov8-air_little_obj_roi/als-1216_sz-32_cls-2_exclude-small_overfit_0/weights/best.pt')

args = {"source": "/home/liuyang/datasets/als_yanshou_tmp/images/", "project": "runs_yolov8-air_little_obj_roi",
        "imgsz": 32, "name": "predict_als-1216_sz-32_cls-2_exclude-small_overfit_0", 'batch': 2048, "save_txt": True, "save_conf": True,
        "conf": 0.001}
results = model.predict(**args)


