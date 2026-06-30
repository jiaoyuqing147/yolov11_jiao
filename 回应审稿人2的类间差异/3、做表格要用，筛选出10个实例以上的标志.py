import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

CSV_PATH = SCRIPT_DIR / "per_class_comparison_all_models.csv"

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

filtered = df[df["Instances"] >= 10].copy()

top10 = filtered.sort_values(
    "AP50-95_improvement_KD_vs_baseline",
    ascending=False
).head(10)

cols = [
    "class_name",
    "Instances",
    "YOLOv11_baseline_AP50-95",
    "ECAFA_YOLO_AP50-95",
    "ECAFA_YOLO_KD_AP50-95",
    "AP50-95_improvement_KD_vs_baseline",
]

table = top10[cols].copy()

table = table.rename(columns={
    "class_name": "Class",
    "YOLOv11_baseline_AP50-95": "YOLOv11",
    "ECAFA_YOLO_AP50-95": "ECAFA-YOLO",
    "ECAFA_YOLO_KD_AP50-95": "ECAFA-YOLO+KD",
    "AP50-95_improvement_KD_vs_baseline": "Δ AP50-95",
})

# 保留三位小数
for col in ["YOLOv11", "ECAFA-YOLO", "ECAFA-YOLO+KD", "Δ AP50-95"]:
    table[col] = table[col].round(3)

out_csv = SCRIPT_DIR / "table_top10_improved_instances_ge10.csv"
table.to_csv(out_csv, index=False, encoding="utf-8-sig")

print(table)
print(f"Saved: {out_csv}")