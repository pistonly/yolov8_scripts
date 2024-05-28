from ultralytics import YOLOWorld
from slice_yolo.slice_models import YOLOWorldSlice


# models = ["yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2", "yolov8x-worldv2",
#           "yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world"]
# models = ["yolov8l-worldv2", "yolov8x-worldv2",
#           "yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world"]
models = ["yolov8l-worldv2"]

for m in models:
    model = YOLOWorldSlice(f'../yolo_models/{m}.pt')

    # args = {"data": "VisDrone.yaml", "project": "yolov8-visdrone",
    #         "device": "0", "batch": 1, 'augment': False, 'imgsz': 2016, 'name': 'val-yolov8l-640train-2016t'}
    args = {"data": "datasets-config/panda_one_label_gega.yaml", "device": "0", "project": "yolo-world_on_panda", "name": f"{m}-1_label_baby-carrage-gega-",
            "batch": 1, "imgsz": 26753, "split": "train", "workers": 1}
    model.set_classes(["baby carriage"])
    metrics = model.val(**args)
