from ultralytics import YOLO


model = YOLO("../../yolo_models/hsh_18_model.pt")
success = model.export(format="onnx")
