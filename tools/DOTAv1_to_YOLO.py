import pandas as pd
from pathlib import Path
import cv2

category_to_labelId = {
    "tennis-court":4,
    "swimming-pool":14,
    "bridge":8,
    "ship":1,
    "storage-tank":2,
    "harbor":7,
    "small-vehicle":10,
    "soccer-ball-field":13,
    "large-vehicle":9,
    "ground-track-field":6,
    "plane":0,
    "baseball-diamond":3,
    "roundabout":12,
    "basketball-court":5,
    "helicopter":11,
}

def obb_to_yolo(x1, y1, x2, y2, x3, y3, x4, y4, category, img_width, img_height):
    # Step 1: Find xmin, ymin, xmax, ymax for the HBB
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)

    # Step 2: Calculate center coordinates, width, and height
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin

    # Step 3: Normalize the values
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    if category not in category_to_labelId:
        category_to_labelId[category] = len(category_to_labelId)
    else:
        category = category_to_labelId[category]

    # Step 4: Format for YOLO
    yolo_format = f"{category} {center_x_norm} {center_y_norm} {width_norm} {height_norm}\n"
    return yolo_format



dota_ds_dir = Path("/home/liuyang/datasets/dota_v1/val")
png_images_dir = dota_ds_dir / "images_png"
obb_dir = dota_ds_dir / "reclabelTxt"
yolo_image_dir = dota_ds_dir / "images"
yolo_label_dir = dota_ds_dir / "labels"
yolo_image_dir.mkdir(exist_ok=True, parents=True)
yolo_label_dir.mkdir(exist_ok=True, parents=True)


for png_path in png_images_dir.iterdir():
    obb_label_path = png_path.with_suffix(".txt")
    obb_label_path = str(obb_label_path).replace("images_png", "reclabelTxt")
    if Path(obb_label_path).is_file():
        img = cv2.imread(str(png_path))
        yolo_img_path = (yolo_image_dir / png_path.name).with_suffix(".jpg")
        cv2.imwrite(str(yolo_img_path), img)
        h, w = img.shape[:2]
        yolo_label_path = yolo_label_dir / Path(obb_label_path).name
        with open(str(yolo_label_path), "w") as f:
            try:
                obb_label = pd.read_csv(obb_label_path, header=None, delimiter=" ")
                for obb_label_i in obb_label.values:
                    yolo_label_i_str = obb_to_yolo(*obb_label_i[0:9], w, h)
                    f.write(yolo_label_i_str)
            except Exception as e:
                print(e)
                print(obb_label_path)

print(category_to_labelId)
with open(str(dota_ds_dir / "label_id.csv"), "w") as f:
    for k, v in category_to_labelId.items():
        f.write(f"{k},{v}\n")




