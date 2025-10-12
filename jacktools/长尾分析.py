# -*- coding: utf-8 -*-
import re
import pandas as pd

# ===== 配置 =====
file_path = r"2.txt"
ratios = (0.2, 0.2, 0.6)  # 20% : 20% : 60%

assert abs(sum(ratios) - 1.0) < 1e-8, "ratios 之和必须为 1.0"

# ===== 读取文件 =====
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# ===== 提取每类的: class, instances, P, R, mAP50, mAP50-95 =====
# 形如：pl80  263  278  0.647  0.788  0.813  0.661
pattern = r"^\s*(\S+)\s+\d+\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
rows = re.findall(pattern, text, flags=re.MULTILINE)

df = pd.DataFrame(rows, columns=["class", "instances", "P", "R", "mAP50", "mAP50-95"])
# 有时日志第一行会有 "all" 汇总项，去掉它
df = df[df["class"] != "all"].copy()

# 类型转换
for c in ["instances", "P", "R", "mAP50", "mAP50-95"]:
    df[c] = df[c].astype(float)

# ===== 排序（按实例数降序）=====
df_sorted = df.sort_values("instances", ascending=False).reset_index(drop=True)

# ===== 按 20% : 20% : 60% 划分 =====
n = len(df_sorted)
g1 = int(n * ratios[0])
g2 = int(n * ratios[1])
# 尾部用剩下的，保证三组之和等于 n
g3 = n - g1 - g2

top_df = df_sorted.iloc[:g1]
mid_df = df_sorted.iloc[g1:g1+g2]
tail_df = df_sorted.iloc[g1+g2:]

def summarize(part):
    return {
        "类别数": len(part),
        "总框体数": int(part["instances"].sum()),
        "平均P": part["P"].mean(),
        "平均R": part["R"].mean(),
        "平均mAP50": part["mAP50"].mean(),
        "平均mAP50-95": part["mAP50-95"].mean(),
        "框体占比(%)": part["instances"].sum() / df_sorted["instances"].sum() * 100,
    }

top_sum  = summarize(top_df)
mid_sum  = summarize(mid_df)
tail_sum = summarize(tail_df)

summary_df = pd.DataFrame([top_sum, mid_sum, tail_sum], index=[f"前{int(ratios[0]*100)}%", f"中{int(ratios[1]*100)}%", f"后{int(ratios[2]*100)}%"])
print("📊 性能分组统计（按实例数降序 & 20:20:60 划分）:")
print(summary_df.round(4))

# 可选：保存
summary_df.to_csv("group_performance_summary_20_20_60.csv", encoding="utf-8-sig")

# 可选：同时打印每组前若干类（便于核对）
print("\n✅ 前组 Top 10 类(按实例数):")
print(top_df[["class","instances","P","R","mAP50","mAP50-95"]].head(10).to_string(index=False))
