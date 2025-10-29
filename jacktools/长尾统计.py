import os
import matplotlib.pyplot as plt
from collections import Counter

# === 配置路径 ===
base_dir = r"E:\DataSets\tt100k_2021\yolojack\labels"
sets = ["train", "val", "test"]
classes_path = os.path.join(base_dir, "classes.txt")

# === 读取类别名 ===
with open(classes_path, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

num_classes = len(class_names)
counter = Counter()

# === 统计每个类别的标注框数量 ===
for subset in sets:
    label_dir = os.path.join(base_dir, subset)
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(label_dir, file), "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    cls = int(line.split()[0])
                    counter[cls] += 1

# === 转换为列表并排序（从多到少） ===
sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
sorted_classes = [c[0] for c in sorted_counts]
sorted_values = [c[1] for c in sorted_counts]

# === 新增：仅保留前 2/3 的类别 ===
keep_num = int(len(sorted_classes) * 2 / 3)  # ← 新增：计算保留的类别数量
sorted_classes = sorted_classes[:keep_num]   # ← 新增：截取前 2/3 类别
sorted_values = sorted_values[:keep_num]     # ← 新增：截取前 2/3 对应数量

# === 绘图 ===
plt.figure(figsize=(14, 6))
plt.bar(range(len(sorted_classes)), sorted_values, color='royalblue', edgecolor='black')

# === X轴标签控制（可选）===
# 方案1：只显示编号（推荐）
plt.xticks(range(len(sorted_classes)), sorted_classes, rotation=90, fontsize=8)

# 方案2：如果想显示名称，请取消下行注释：
# plt.xticks(range(len(sorted_classes)), [class_names[i] for i in sorted_classes], rotation=90, fontsize=6)

# plt.title("Bounding Box Count per Class (Sorted by Frequency)", fontsize=14)
plt.xlabel("Class Index ", fontsize=24)  # ← 修改：标题更明确
plt.ylabel("Number of Bounding Boxes", fontsize=24)

# === 优化布局并保存 ===
plt.tight_layout()
save_path = os.path.join(base_dir, "bbox_count_sorted_top2_3.png")  # ← 修改：新文件名
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ 已保存条形图到: {save_path}")
