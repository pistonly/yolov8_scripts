import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO


model = YOLO('yolov8n.pt')

add_wandb_callback(model, enable_model_checkpointing=True)
args = {"data": "VisDrone.yaml", "epochs": 300, "project": "yolov8-visdrone"}
results = model.train(**args)

wandb.finish()
