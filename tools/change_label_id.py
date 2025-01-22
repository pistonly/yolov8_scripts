from pathlib import Path


from_dir = Path("/home/liuyang/datasets/als_uav_ds_augment")
to_dir = Path("/home/liuyang/datasets/als_uav_ds_augment")
to_dir.mkdir(exist_ok=True, parents=True)

from_label = "0"
to_label = "1"

label_paths = sorted(from_dir.glob("**/*.txt"))
for label_f in label_paths:
    relative_label_path = label_f.relative_to(from_dir)
    new_label_f = to_dir / relative_label_path

    lines_new = []
    with open(str(label_f), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split()
            if int(float(line_list[0])) == int(from_label):
                line_list[0] = to_label
            lines_new.append(" ".join(line_list))

    with open(str(new_label_f), "w") as f:
        for line in lines_new:
            f.write(line)
            f.write("\n")

