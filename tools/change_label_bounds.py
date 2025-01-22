from pathlib import Path


from_dir = Path("/home/liuyang/datasets/als_yanshou_tmp/labels")
to_dir = Path("/home/liuyang/datasets/als_yanshou_tmp/labels_new")
to_dir.mkdir(exist_ok=True, parents=True)

for label_f in from_dir.iterdir():
    lines_new = []
    with open(str(label_f), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split()
            for i in range(1, 5):
                if float(line_list[i]) > 1:
                    line_list[i] = "1"
            lines_new.append(" ".join(line_list))
    new_label_f = to_dir / label_f.name
    with open(str(new_label_f), "w") as f:
        for line in lines_new:
            f.write(line)
            f.write("\n")

