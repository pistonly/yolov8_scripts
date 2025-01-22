from ultralytics import YOLOWorld


# models = ["yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2", "yolov8x-worldv2",
#           "yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world"]
# models = ["yolov8l-worldv2", "yolov8x-worldv2",
#           "yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world"]
models = ["yolov8x-worldv2"]

for m in models:
    model = YOLOWorld(f'../yolo_models/{m}.pt')

    args = {"device": "0", "project": "results/yolo-world_on_panda", "name": f"predict-panda-{m}-prompt_red-color-flag",
            "batch": 1, "imgsz": 1280, "split": "train", "source": "/home/liuyang/datasets/PANDA/YOLO/train_one_label_1280/train.txt",
            "save": True, "agnostic_nms": True}
    # model.set_classes(["baby carriage", "two-wheel carriage"])
    # model.set_classes(["red flag", "red object"])
    # model.set_classes(["stroller", "baby carriage"])
    # model.set_classes(["Chinese character"])
    # model.set_classes(["red car", "white car", "black car"])
    # model.set_classes(["flag", "red flag"])
    model.set_classes(["red color flag"])


    model.predict(**args)
