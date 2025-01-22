from PIL import Image
from pathlib import Path

# 读取WebP格式的图片
keep_name = True
img_dir = Path("/home/liuyang/datasets/haimasi/from_zhang")
output_dir = Path("/home/liuyang/datasets/haimasi/from_zhang_jpgs/")
output_dir.mkdir(parents=True, exist_ok=True)
for i, img_path in enumerate(img_dir.iterdir()):
    print(img_path)
    image = Image.open(str(img_path))
    # 将图片转换为其他格式并保存（例如JPEG）
    if keep_name:
        img_stem = img_path.stem
        image.convert('RGB').save(str(output_dir / f"{int(img_stem):04d}.jpg"), 'JPEG')
    else:
        image.convert('RGB').save(str(output_dir / f"{i:04d}.jpg"), 'JPEG')
