# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

# ===== Global style (paper-ready) =====
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.size'] = 12

# ===== Data =====
groups = ["Head (Top 20%)", "Medium (Middle 20%)", "Tail (Bottom 60%)"]

# YOLO11 baseline
yolo11_recall  = [0.6495, 0.4401, 0.1821]
yolo11_map5095 = [0.4720, 0.3170, 0.1401]

# Your improved model (Ours)
ours_recall  = [0.8006, 0.6482, 0.3241]
ours_map5095 = [0.6224, 0.5115, 0.2946]

# ===== Layout parameters =====
x = np.arange(len(groups))
bar_width = 0.16                 # width of each bar
gap_between_metrics = 0.18       # spacing between Recall and mAP@0.5:0.95 inside each group

# Centers for the two metric blocks within each group
x_recall = x - bar_width/2 - gap_between_metrics/2
x_map    = x + bar_width/2 + gap_between_metrics/2

# Colors (paper-friendly)
colors = {
    "yolo_recall": "#5B8FF9",  # blue
    "ours_recall": "#91CB74",  # green
    "yolo_map":    "#F6BD16",  # orange
    "ours_map":    "#E8684A",  # red
}

# ===== Plot =====
fig = plt.figure(figsize=(7.6, 4.8))

# Recall bars
b1 = plt.bar(x_recall - bar_width/2, yolo11_recall,  width=bar_width, color=colors["yolo_recall"], alpha=0.85, label='YOLO11 Recall')
b2 = plt.bar(x_recall + bar_width/2, ours_recall,    width=bar_width, color=colors["ours_recall"], alpha=0.85, label='Ours Recall')

# mAP@0.5:0.95 bars
b3 = plt.bar(x_map - bar_width/2,    yolo11_map5095, width=bar_width, color=colors["yolo_map"],   alpha=0.85, label='YOLO11 mAP@0.5:0.95')
b4 = plt.bar(x_map + bar_width/2,    ours_map5095,   width=bar_width, color=colors["ours_map"],   alpha=0.85, label='Ours mAP@0.5:0.95')

# Axis & grid
plt.xticks(x, groups)
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.yticks(np.linspace(0, 1.0, 6))
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Legend
plt.legend(fontsize=10, loc='upper right', frameon=False, ncol=2)

# Title
#plt.title("Comparison of Recall and mAP@0.5:0.95 across Frequency Groups", fontsize=13, pad=10)

# ===== Value labels on top of bars =====
def add_value_labels(bars, fmt="{:.2f}", dy=0.01):
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            h + dy,
            fmt.format(h),
            ha='center', va='bottom', fontsize=10
        )

add_value_labels(b1); add_value_labels(b2)
add_value_labels(b3); add_value_labels(b4)

plt.tight_layout()
plt.savefig("LongTail_Recall_mAP5095_Combined.png", dpi=600, bbox_inches='tight')
plt.savefig("LongTail_Recall_mAP5095_Combined.pdf", dpi=600, bbox_inches='tight')  # vector for print
plt.show()
