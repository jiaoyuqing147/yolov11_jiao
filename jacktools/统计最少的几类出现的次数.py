import os
from collections import Counter

# === 配置路径 ===
base_dir = r"E:\DataSets\tt100k_2021\yolojack\labels"
sets = ["train", "val", "test"]
classes_path = os.path.join(base_dir, "classes.txt")

# === 读取类别名 ===
with open(classes_path, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

counter = Counter()

# === 统计每个类别的标注框数量 ===
for subset in sets:
    label_dir = os.path.join(base_dir, subset)
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(label_dir, file), "r", encoding="utf-8") as f:
            for line in f:
                cls = int(line.split()[0])
                counter[cls] += 1

# === 按出现次数排序 ===
sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)

print("===== Top 10 Frequent Classes =====")
for cls, count in sorted_counts[:10]:
    print(f"{class_names[cls]} : {count}")

print("\n===== Top 10 Rare Classes =====")
for cls, count in sorted_counts[-10:]:
    print(f"{class_names[cls]} : {count}")
