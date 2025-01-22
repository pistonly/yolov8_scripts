from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from pathlib import Path


# Define an inference source
source = "/home/liuyang/datasets/PANDA/YOLO_1280/images/03_Train_Station Square/IMG_03_30.jpg"

# Create a FastSAM model
model = FastSAM("../yolo_models/FastSAM-s.pt")  # or FastSAM-x.pt

# Run inference on an image
everything_results = model(source, device="cuda:0", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

# Prepare a Prompt Process object
prompt_process = FastSAMPrompt(source, everything_results, device="cuda:0")

# Text prompt
ann = prompt_process.text_prompt(text="a photo of a chinese character")

prompt_process.plot(annotations=ann, output="./results/fast-sam_panda_chinese-character/")
