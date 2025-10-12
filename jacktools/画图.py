# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 1) 模型 CSV 路径与名称
# ===============================
# csv_paths = [
#     r'../runsMTSD/yolo11_train/exp/results.csv',
#     r'../runsMTSD/yolo11-FASFFHead_P234_train/exp/results.csv',
#     r'../runsMTSD/yolo11-FASFFHead_P234_RCSOSA_ciou_bce_train/exp/results.csv',
#     r'../runsMTSD/yolo11-FASFFHead_P234_RCSOSA_wiou_bce_train/exp/results.csv',
#     r'../runsMTSD/yolo11-FASFFHead_P234_RCSOSA_wiou_bce_distillation/exp/results.csv',
# ]

csv_paths = [
    r'../runsTT100k130/yolo11_train/exp/results.csv',
    r'../runsTT100k130/yolo11-FASFFHead_P234_train/exp/results.csv',
    r'../runsTT100k130/yolo11-FASFFHead_P234_RCSOSA_ciou_bce_train/exp/results.csv',
    r'../runsTT100k130/yolo11-FASFFHead_P234_RCSOSA_wiou_bce_train/exp/results.csv',
    r'../runsTT100k130/yolo11-FASFFHead_P234_RCSOSA_wiou_bce_distillation/exp/results.csv',
]

model_names = [
    'YOLO11n',
    'YOLO11n + STS',
    'YOLO11n + STS + RCSOSA',
    'YOLO11n + STS + RCSOSA + WIOU',
    'YOLO11n + STS + RCSOSA + WIOU + Distill',
]

# ===============================
# 2) 通用参数
# ===============================
drop_last_epoch = True
smooth_window = 5
mark_best = True
dpi = 600
save_dir = r'../runsTT100k130'

plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
linewidth = 1.2

# ===============================
# 3) 工具函数
# ===============================
def load_and_prepare(path, cols):
    """读取 CSV → 去掉最后 epoch → 平滑数据"""
    if not os.path.exists(path):
        raise FileNotFoundError(f'未找到文件: {path}')
    df = pd.read_csv(path)
    for c in cols + ['epoch']:
        if c not in df.columns:
            raise ValueError(f'{path} 缺少列: {c}')
    if drop_last_epoch:
        max_ep = df['epoch'].max()
        df = df[df['epoch'] < max_ep].reset_index(drop=True)
    smooth = {c: df[c].rolling(window=smooth_window, center=True, min_periods=1).median() for c in cols}
    return df['epoch'], df, smooth

def plot_metric(metric_col, title, ylabel, outfile, pick='max'):
    """通用绘图函数：画一张指标曲线（mAP / Recall / Loss）"""
    plt.figure(figsize=(8, 6))
    for path, name, color in zip(csv_paths, model_names, colors):
        epoch, df, sm = load_and_prepare(path, [metric_col])
        y = sm[metric_col]
        plt.plot(epoch, y, color=color, linewidth=linewidth, label=name)

        if mark_best:
            idx = y.idxmax() if pick == 'max' else y.idxmin()
            plt.scatter([epoch[idx]], [y[idx]], color=color, s=18, zorder=3)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # ---- 原本的：标题在图外（保留为注释，随时可还原）----
    # plt.title(title, fontsize=14, fontweight='bold')

    # ✅ 新增：把标题放到图内部左上角
    ax = plt.gca()
    ax.text(0.02, 0.95, title,
            transform=ax.transAxes,
            fontsize=10, fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=3))

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    outpath = os.path.join(save_dir, outfile)
    plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f'✅ 保存：{outpath}')

# ===============================
# 4) Recall 曲线
# ===============================
def plot_recall():
    plt.figure(figsize=(8,6))
    for path, name, color in zip(csv_paths, model_names, colors):
        epoch, df, sm = load_and_prepare(path, ['metrics/recall(B)', 'metrics/mAP50-95(B)'])
        y = sm['metrics/recall(B)']
        plt.plot(epoch, y, color=color, linewidth=linewidth, label=name)

        if mark_best:
            idx = sm['metrics/mAP50-95(B)'].idxmax()
            plt.scatter([epoch[idx]], [y[idx]], color=color, s=18, zorder=3)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Recall', fontsize=12)

    # ---- 原本的：标题在图外（保留为注释）----
    # plt.title('Recall Comparison of YOLO11n Variants on TT100K Dataset',
    #           fontsize=14, fontweight='bold')

    # ✅ 新增：把标题放到图内部左上角
    ax = plt.gca()
    ax.text(0.02, 0.95,
            'Recall Comparison of YOLO11n Variants on TT100K Dataset',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=3))

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    outpath = os.path.join(save_dir, 'recall_comparison_TT100K.png')
    plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f'✅ 保存：{outpath}')

# ===============================
# 5) Total Validation Loss 曲线
# ===============================
def plot_total_val_loss():
    plt.figure(figsize=(8,6))
    for path, name, color in zip(csv_paths, model_names, colors):
        epoch, df, sm = load_and_prepare(path, ['val/box_loss', 'val/cls_loss', 'val/dfl_loss'])
        total_loss = sm['val/box_loss'] + sm['val/cls_loss'] + sm['val/dfl_loss']
        plt.plot(epoch, total_loss, color=color, linewidth=linewidth, label=name)
        if mark_best:
            idx = total_loss.idxmin()
            plt.scatter([epoch[idx]], [total_loss[idx]], color=color, s=18, zorder=3)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Validation Loss', fontsize=12)

    # ---- 原本的：标题在图外（保留为注释）----
    # plt.title('Total Validation Loss of YOLO11n Variants on TT100K Dataset',
    #           fontsize=14, fontweight='bold')

    # ✅ 新增：把标题放到图内部左上角
    ax = plt.gca()
    ax.text(0.02, 0.95,
            'Total Validation Loss of YOLO11n Variants on TT100K Dataset',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=3))

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    outpath = os.path.join(save_dir, 'total_val_loss_TT100K.png')
    plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.show()
    print(f'✅ 保存：{outpath}')

# ===============================
# 6) 执行所有图表绘制
# ===============================
if __name__ == '__main__':
    plot_recall()
    plot_metric('metrics/mAP50(B)',
                'mAP@0.5 Comparison of YOLO11n Variants on TT100K Dataset',
                'mAP@0.5', 'map50_comparison_TT100K.png', pick='max')
    plot_metric('metrics/mAP50-95(B)',
                'mAP@0.5:0.95 Comparison of YOLO11n Variants on TT100K Dataset',
                'mAP@0.5:0.95', 'map50_95_comparison_TT100K.png', pick='max')
    plot_total_val_loss()
