from ultralytics import YOLO
from pathlib import Path

# model = YOLO("/home/liuyang/Documents/YOLO/yolov8_scripts/yolov8-air_little_obj_roi/yolov8n7/weights/best.pt")
model = YOLO("./yolov8_yanshou-20240922-0930-babiao/yolov8n-continue-train-val-no-overlap/weights/best.pt")
# model = YOLO('../yolo_models/yolov8n_sod4bird_finetune.pt')

# for source_stem in ["20241017111130963-12-2-main", "20241017112031093-100-1-main", "20241017112331240-32-1-main", "MAX_0014"]:
#     source = str(Path("/home/liuyang/datasets/damao") / source_stem)
#     args = {"source": source, "project": "yolov8_yanshou-20240922-0930-babiao",
#             "imgsz": 2560, "name": f"pred_{source_stem}_conf-0.01_", 'batch': 1, 'conf': 0.01, "save": True, 'line_width': 1}
#     results = model.predict(**args)

for source_stem in ["babiao/images/"]:
    source = str(Path("/home/liuyang/datasets/") / source_stem)
    args = {"source": source, "project": "yolov8_yanshou-20240922-0930-babiao",
            "imgsz": 2560, "name": f"pred_babiao_conf-0.01_", 'batch': 1, 'conf': 0.01, "save": True, 'line_width': 1}
    results = model.predict(**args)

