import os

# 数据集标注路径
base_dir = r"E:\DataSets\MTSD\yolo54\labels"

file_counts = []

# 遍历 train / val / test 文件夹
for split in ["train", "val", "test"]:
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(split_dir):
        continue
    for fname in os.listdir(split_dir):
        if fname.endswith(".txt"):
            fpath = os.path.join(split_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # 每一行就是一个目标
            count = len([l for l in lines if l.strip()])
            file_counts.append((fpath, count))

# 排序，取前5个
top5 = sorted(file_counts, key=lambda x: x[1], reverse=True)[:5]

print("目标数量最多的前5个文件：")
for fpath, count in top5:
    print(f"{fpath} -> {count} objects")
