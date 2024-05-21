from ultralytics import YOLO
from pathlib import Path

model_dir = Path("../../yolo_models")
# Load the YOLOv8 model
model = YOLO(str(model_dir / "hsh_yolov8n_first.pt"))

# Export the model to TensorRT format
model.export(format="engine", half=True, imgsz=640)  # creates 'yolov8n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO(str(model_dir / "hsh_yolov8n_first.engine"))

# Run inference
results = tensorrt_model("./data/test1-0_20240514-152234-0__1.jpg")
