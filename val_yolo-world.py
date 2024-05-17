from ultralytics import YOLOWorld


# models = ["yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2", "yolov8x-worldv2",
#           "yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world"]
# models = ["yolov8l-worldv2", "yolov8x-worldv2",
#           "yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world"]
models = ["yolov8s-worldv2-finetune-visdrone-1280"]

for m in models:
    model = YOLOWorld(f'../yolo_models/{m}.pt')

    # args = {"data": "VisDrone.yaml", "project": "yolov8-visdrone",
    #         "device": "0", "batch": 1, 'augment': False, 'imgsz': 2016, 'name': 'val-yolov8l-640train-2016t'}
    args = {"data": "datasets-config/silu__dataset_4k_2.yaml", "device": "6", "project": "yolo-world_test", "name": f"silu__dataset_4k__{m}-0",
            "batch": 1, "imgsz": 3840, "split": "train", }
    model.set_classes(["person", "car", "truck", "boat", "airplane", "train", "bicycle", "tricycle"])
    metrics = model.val(**args)
