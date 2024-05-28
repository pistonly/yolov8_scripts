from ultralytics import YOLOWorld


# models = ["yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2", "yolov8x-worldv2",
#           "yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world"]
models = ["yolov8s-worldv2-finetune-visdrone-1280", "yolov8s-worldv2",]

for m in models:
    model = YOLOWorld(f'../yolo_models/{m}.pt')

    # args = {"data": "VisDrone.yaml", "project": "yolov8-visdrone",
    #         "device": "0", "batch": 1, 'augment': False, 'imgsz': 2016, 'name': 'val-yolov8l-640train-2016t'}
    args = {"data": "datasets-config/silu__dataset_4k_2.yaml", "device": "0", "project": "yolo-world_test", "name": f"predict__dataset_4k__sample__{m}-0",
            "batch": 1, "imgsz": 3840, "split": "train", "source": "/home/liuyang/datasets/qiyuan/dataset_4k/sample/",
            "save": True}
    model.set_classes(["person", "car", "truck", "boat", "airplane", "train", "bicycle", "tricycle"])
    model.predict(**args)
