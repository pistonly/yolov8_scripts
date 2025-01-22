from ultralytics import YOLO

# model = YOLO('../yolo_models/yolov8n_DOTAv1-1024.pt')
# model = YOLO('/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yanshou_laixi_1122/batch-36_sz-1920_freeze-10_lr0-0005_2/weights/best.pt')
model = YOLO("/home/liuyang/Documents/YOLO/yolov8_scripts/runs_yanshou_laixi_1122/batch-36_sz-1920_freeze-10_lr0-0005_2/weights/backup-step-10/best.pt")

# source = f"/home/liuyang/Downloads/yanshou_laixi/1122/1122_0706_images/11-2_selected"
# source = f"/home/liuyang/Downloads/yanshou_laixi/1122/1122_0650_images/20241122065019276-100-1-main/"
# source = "/home/liuyang/Downloads/yanshou_laixi/1122/1122_0650_images/20241122065019296-21-1-main/"
# source = "/home/liuyang/Downloads/yanshou_laixi/1122/1122_0650_images/20241122065019296-22-1-main"
source = "/home/liuyang/Downloads/yanshou_laixi/1122/1122_0642_images/22-1-sel-2/"
args = {"source": source, "project": "runs_yanshou_laixi",
        "imgsz": 1920, "name": f"0642-22-1-selected-2__model-laixi-1122-model-2-step-10-conf-01_", 'batch': 5,
        "save_frames": True, "save": True, "conf": 0.1, "stream": True
        }
results = model.predict(**args)

# # print(results)
for r in results:
    pass
    # boxes = r.boxes  # Box object used for boundary box output
    # masks = r.masks  # Mask object used to split mask output
    # probs = r.probs  # Category probability for classification output

