from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from pathlib import Path


source_dir = Path("tools/results/panda_baby-carriage_1024")

for img in source_dir.iterdir():
    if img.suffix not in ['.jpg', '.pnb']:
        continue
    # Define an inference source
    source = str(img)

    # Create a FastSAM model
    model = FastSAM("../yolo_models/FastSAM-s.pt")  # or FastSAM-x.pt

    # Run inference on an image
    everything_results = model(source, device="cuda:0", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

    # Prepare a Prompt Process object
    prompt_process = FastSAMPrompt(source, everything_results, device="cuda:0")

    # Text prompt
    ann = prompt_process.text_prompt(text="a photo of a chinese character")

    prompt_process.plot(annotations=ann, output="./results/fast-sam_panda_chinese-character/")
