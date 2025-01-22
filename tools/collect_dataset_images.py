from pathlib import Path
import shutil

# 定义源目录和目标目录
from_dir = Path("/home/liuyang/datasets/air_little_obj_als/")
to_dir = Path("./results/air_little_obj_als")
to_dir.mkdir(exist_ok=True, parents=True)

# 创建目标子目录
images_dir = to_dir / "images"
labels_dir = to_dir / "labels"
images_dir.mkdir(exist_ok=True)
labels_dir.mkdir(exist_ok=True)

# 获取所有 jpg 文件和 labels/*.txt 文件
jpg_files = list(from_dir.rglob("*.jpg"))
txt_files = list(from_dir.rglob("labels/*.txt"))

# 定义函数：生成重命名文件的路径
def get_new_name(file: Path, base_dir: Path, target_dir: Path, is_label: bool):
    # 获取相对路径并将路径分隔符替换为'-'
    relative_path = file.relative_to(base_dir).with_suffix('')
    path_parts = list(relative_path.parts)
    if is_label:
        path_parts[-2] = "images"

    new_name = '-'.join(path_parts) + file.suffix
    return target_dir / new_name

# 拷贝 jpg 文件到目标目录
for jpg_file in jpg_files:
    new_path = get_new_name(jpg_file, from_dir, images_dir, False)
    shutil.copy(jpg_file, new_path)

# 拷贝 txt 文件到目标目录
for txt_file in txt_files:
    new_path = get_new_name(txt_file, from_dir, labels_dir, True)
    shutil.copy(txt_file, new_path)

print("拷贝完成！")
