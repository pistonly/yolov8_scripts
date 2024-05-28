from ultralytics import YOLOWorld


# models = ["yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2", "yolov8x-worldv2",
#           "yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world"]
# models = ["yolov8l-worldv2", "yolov8x-worldv2",
#           "yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world"]
models = ["yolov8l-worldv2"]

for m in models:
    model = YOLOWorld(f'../yolo_models/{m}.pt')

    # args = {"data": "VisDrone.yaml", "project": "yolov8-visdrone",
    #         "device": "0", "batch": 1, 'augment': False, 'imgsz': 2016, 'name': 'val-yolov8l-640train-2016t'}
    args = {"data": "datasets-config/panda_one_label_1280.yaml", "device": "0", "project": "yolo-world_on_panda", "name": f"{m}-1_label_baby_carriage-",
            "batch": 1, "imgsz": 1280, "split": "train", }
    model.set_classes(["baby carriage"])
    metrics = model.val(**args)
