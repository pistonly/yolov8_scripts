import glob
import os
import numpy as np

def collect_jpg_files_glob(directory_path):
    # 构建查找.jpg文件的路径模式
    pattern = os.path.join(directory_path, '**', '*.txt')
    
    # 使用glob.glob()查找所有匹配的文件，recursive=True允许搜索子目录
    jpg_files = glob.glob(pattern, recursive=True)
    
    return jpg_files

# 指定要遍历的文件夹路径
directory_path = '/home/liuyang/datasets/database0/secondpc'
directory_len = len(directory_path)

# 调用函数并打印结果
label_files = collect_jpg_files_glob(directory_path)
label_files = [f_path[directory_len:] for f_path in label_files]
images_files = [f'''.{f_path.replace("labels", "images").replace(".txt", ".jpg")}''' for f_path in label_files]
# print(images_files)
with open("train.txt", "w") as f:
    for img_p in images_files:
        f.write(img_p)
        f.write("\n")

