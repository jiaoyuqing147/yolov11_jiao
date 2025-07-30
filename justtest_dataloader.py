import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import yaml

# ✅ 修改这里：设置路径和参数
yaml_path = 'ultralytics/cfg/datasets/tt100k_chu.yaml'  # ← 替换成你的数据集配置
max_cls_id = 232  # 最大类别数（从0开始，最大值为 231）

# ✅ YAML 加载函数（兼容你本地的 yolov11_jiao 项目）
def yaml_load(file):
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)

def check_yaml(file):
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"YAML file not found: {file}")
    return str(file)

# ✅ 读取 yaml 文件
yaml_file = check_yaml(yaml_path)
data_cfg = yaml_load(yaml_file)
train_dir = data_cfg['train']
label_dir = data_cfg.get('labels', train_dir.replace('images', 'labels'))

# ✅ 合法图像后缀
img_suffixes = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ✅ 遍历图像并验证
image_files = [f for f in Path(train_dir).rglob('*') if f.suffix.lower() in img_suffixes]
bad_images = []

print(f'\n🔍 正在检查 {len(image_files)} 张图像是否损坏...')
for img_path in tqdm(image_files):
    try:
        with Image.open(img_path) as im:
            im.verify()
    except Exception as e:
        print(f'❌ 损坏图像: {img_path}，原因: {e}')
        bad_images.append(img_path)

# ✅ 删除坏图及其标签
for bad in bad_images:
    try:
        os.remove(bad)
        label_path = Path(label_dir) / (bad.stem + '.txt')
        if label_path.exists():
            os.remove(label_path)
        print(f'🗑️ 已删除: {bad}')
    except Exception as e:
        print(f'⚠️ 删除失败: {bad}, {e}')

# ✅ 遍历标签文件，检查空文件和越界
label_files = list(Path(label_dir).rglob('*.txt'))
bad_labels = []

print(f'\n🔍 正在检查 {len(label_files)} 个标签文件...')
for label_path in tqdm(label_files):
    try:
        if os.path.getsize(label_path) == 0:
            bad_labels.append(label_path)
            print(f'⚠️ 空标签: {label_path}')
            continue

        with open(label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 1:
                bad_labels.append(label_path)
                print(f'❌ 标签格式错误: {label_path}')
                break
            cls_id = int(float(parts[0]))
            if cls_id < 0 or cls_id >= max_cls_id:
                bad_labels.append(label_path)
                print(f'❌ 标签越界: {label_path}，类别 ID={cls_id}')
                break
    except Exception as e:
        bad_labels.append(label_path)
        print(f'❌ 标签文件异常: {label_path}, {e}')

# ✅ 删除坏标签
for bad in bad_labels:
    try:
        os.remove(bad)
        print(f'🗑️ 已删除标签: {bad}')
    except Exception as e:
        print(f'⚠️ 删除失败: {bad}, {e}')

print(f'\n✅ 完成：共移除 {len(bad_images)} 张损坏图像，{len(bad_labels)} 个异常标签文件\n')