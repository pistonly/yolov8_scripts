from ultralytics import YOLO
import wandb
wandb.login()

model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)

data = "/home/liuyang/datasets/quantaeye/chanliu_spaceA-18_cls/splitted_cls_dataset"

args = {"data": data, "epochs": 300, "project": "chanliu_tianshui",
        "imgsz": 640, "name": "yolo11n-cls-b32-e300_", 'batch': 32, "workers": 8}

results = model.train(**args)
