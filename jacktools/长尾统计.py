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

# === 直接使用类别名称作为 x 轴标签 ===
x_labels = [class_names[cls] for cls in sorted_classes]  # 根据类别索引找到类别名称

# === 绘图 ===
plt.figure(figsize=(14, 6))
bars = plt.bar(range(len(sorted_classes)), sorted_values, color='royalblue', edgecolor='black')

# === X轴标签控制（使用类别名称）===
plt.xticks(range(len(sorted_classes)), x_labels, rotation=90, fontsize=12)

# === 在每个柱状图上添加标签 ===
# for bar in bars:
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02,  # 在柱子顶部显示
#              f'{int(yval)}', ha='center', va='bottom', fontsize=10)

# === 修改Y轴标签 ===
plt.xlabel("Class Name", fontsize=24)
plt.ylabel("Number of Instances", fontsize=24)

# === 优化布局并保存 ===
plt.tight_layout()
save_path = os.path.join(base_dir, "bbox_count_sorted_top2_3_with_labels.png")  # ← 修改：新文件名
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ 已保存条形图到: {save_path}")
