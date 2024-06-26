from ultralytics import YOLOWorld


# models = ["yolov8s-worldv2", "yolov8m-worldv2", "yolov8l-worldv2", "yolov8x-worldv2",
#           "yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world"]
# models = ["yolov8l-worldv2", "yolov8x-worldv2",
#           "yolov8s-world", "yolov8m-world", "yolov8l-world", "yolov8x-world"]
models = ["yolov8l-worldv2"]

for m in models:
    model = YOLOWorld(f'../yolo_models/{m}.pt')

    args = {"device": "0", "project": "results/yolo-world_on_panda", "name": f"predict-panda-{m}-1_label_baby_carriage-prompt_two_wheel_carriage",
            "batch": 1, "imgsz": 1280, "split": "train", "source": "/home/liuyang/datasets/PANDA/YOLO/train_one_label_1280/train.txt",
            "save": True}
    model.set_classes(["baby carriage", "two-wheel carriage"])
    model.predict(**args)
