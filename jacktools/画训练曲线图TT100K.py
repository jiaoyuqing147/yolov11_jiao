# -*- coding: utf-8 -*-
import os
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# 字体设置（避免乱码）
# ===============================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ===============================
# 1) 模型 CSV 路径与名称
# ===============================
csv_paths = [
    r'../runsTT100k130/yolo11_train200/exp/results.csv',
    r'../runsTT100k130/yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train200/exp/results.csv',
    r'../runsTT100k130/yolo11-FASFFHead_P234_OECSOSAInterleave_ciou_bce_train_distillation/exp/results.csv',
]

model_names = [
    'YOLOv11',
    'ECAFA-YOLO (Ours)',
    'ECAFA-YOLO + KD (Ours)',
]

# ===============================
# 2) 通用参数
# ===============================
drop_last_epoch = False
smooth_window = 5
mark_best = True
dpi = 600
save_dir = r'../runsTT100k130'

plt.style.use('seaborn-v0_8-whitegrid')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
linewidth = 1.5

# ===============================
# 3) 保存函数（核心改进）
# ===============================
def save_figure(base_name):
    os.makedirs(save_dir, exist_ok=True)

    svg_path = os.path.join(save_dir, base_name + '.svg')
    png_path = os.path.join(save_dir, base_name + '.png')

    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.savefig(png_path, dpi=dpi, bbox_inches='tight')

    print(f'✅ 保存 SVG: {svg_path}')
    print(f'✅ 保存 PNG: {png_path}')


# ===============================
# 4) 数据处理
# ===============================
def load_and_prepare(path, cols):
    if not os.path.exists(path):
        raise FileNotFoundError(f'未找到文件: {path}')

    df = pd.read_csv(path)

    for c in cols + ['epoch']:
        if c not in df.columns:
            raise ValueError(f'{path} 缺少列: {c}')

    if drop_last_epoch:
        max_ep = df['epoch'].max()
        df = df[df['epoch'] < max_ep].reset_index(drop=True)

    smooth = {
        c: df[c].rolling(window=smooth_window, center=True, min_periods=1).median()
        for c in cols
    }

    return df['epoch'], df, smooth


# ===============================
# 5) 通用指标曲线
# ===============================
def plot_metric(metric_col, title, ylabel, outfile, pick='max'):
    plt.figure(figsize=(3.5, 3.5))

    for path, name, color in zip(csv_paths, model_names, colors):
        epoch, df, sm = load_and_prepare(path, [metric_col])
        y = sm[metric_col]

        plt.plot(epoch, y, color=color, linewidth=linewidth, label=name)

        if mark_best:
            idx = y.idxmax() if pick == 'max' else y.idxmin()
            plt.scatter([epoch[idx]], [y[idx]], color=color, s=25, zorder=3)

    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel(ylabel, fontsize=11)

    # 标题放图内（论文推荐）
    ax = plt.gca()
    ax.text(0.02, 0.95, title,
            transform=ax.transAxes,
            fontsize=10, fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()

    save_figure(outfile)
    plt.show()


# ===============================
# 6) Recall 曲线
# ===============================
def plot_recall():
    plt.figure(figsize=(3.5, 3.5))

    for path, name, color in zip(csv_paths, model_names, colors):
        epoch, df, sm = load_and_prepare(path, ['metrics/recall(B)', 'metrics/mAP50-95(B)'])
        y = sm['metrics/recall(B)']

        plt.plot(epoch, y, color=color, linewidth=linewidth, label=name)

        if mark_best:
            idx = sm['metrics/mAP50-95(B)'].idxmax()
            plt.scatter([epoch[idx]], [y[idx]], color=color, s=25, zorder=3)

    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Recall', fontsize=11)

    ax = plt.gca()
    ax.text(0.02, 0.95,
            'Recall',
            transform=ax.transAxes,
            fontsize=10, fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()

    save_figure('recall_comparison_TT100K')
    plt.show()


# ===============================
# 7) Loss 曲线
# ===============================
def plot_total_val_loss():
    plt.figure(figsize=(3.5, 3.5))

    for path, name, color in zip(csv_paths, model_names, colors):
        epoch, df, sm = load_and_prepare(path, ['val/box_loss', 'val/cls_loss', 'val/dfl_loss'])

        total_loss = sm['val/box_loss'] + sm['val/cls_loss'] + sm['val/dfl_loss']

        plt.plot(epoch, total_loss, color=color, linewidth=linewidth, label=name)

        if mark_best:
            idx = total_loss.idxmin()
            plt.scatter([epoch[idx]], [total_loss[idx]], color=color, s=25, zorder=3)

    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Total Validation Loss', fontsize=11)

    ax = plt.gca()
    ax.text(0.02, 0.95,
            'Validation Loss Comparison on TT100K',
            transform=ax.transAxes,
            fontsize=10, fontweight='bold',
            va='top', ha='left',
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=9)
    plt.tight_layout()

    save_figure('total_val_loss_TT100K')
    plt.show()


# ===============================
# 8) 主函数
# ===============================
if __name__ == '__main__':
    plot_recall()

    plot_metric('metrics/mAP50-95(B)',
                'mAP@50–95',
                'mAP@50–95',
                'map50_95_comparison_TT100K',
                pick='max')

    plot_total_val_loss()