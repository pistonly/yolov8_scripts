from ultralytics import YOLO
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Load YOLO model once (shared among processes if possible)
model = YOLO('/home/liuyang/Documents/YOLO/yolov8_scripts/yanshou_vehicle/yolov8n-20240922-0930-vehicle-640_/weights/best.pt')

# Define prediction function
def process_video(video_dir):
    source = str(video_dir)
    args = {
        "source": source,
        "project": "yanshou_laixi_1122_0650",
        "imgsz": 1920,
        "name": f"{video_dir.name}",
        "batch": 1,
        "save_frames": True,
        "save": True,
        "save_txt": True,
        "show_labels": True,
        "conf": 0.1,
        "line_width": 2,
        "stream": True
    }
    results = model.predict(**args)
    # Ensure all results are processed
    for r in results:
        pass
    print(f"Processed: {video_dir.name}")

# Get list of video directories
all_channel_images = Path("/home/liuyang/Downloads/yanshou_laixi/1122/1122_0650_images")
video_dirs = [vd for vd in all_channel_images.iterdir() if vd.is_dir()]

# Parallel execution
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=10) as executor:
        executor.map(process_video, video_dirs)
