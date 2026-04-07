import cv2
import numpy as np
import os

def letterbox(img, new_shape=640, color=(114, 114, 114)):
    shape = img.shape[:2]  # h, w

    # 计算缩放比例
    r = min(new_shape / shape[0], new_shape / shape[1])

    # 计算新尺寸
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))

    # resize
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # padding
    dw = new_shape - new_unpad[0]
    dh = new_shape - new_unpad[1]

    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2

    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)

    return img


# ===== 输入输出路径 =====
input_dir = r"F:\DataSets\resultTT100k130test\multi_model_comparenew\TopK_vis"
output_dir = r"F:\DataSets\resultTT100k130test\multi_model_comparenew\TopK_vis_640"

# 创建输出文件夹（不存在就创建）
os.makedirs(output_dir, exist_ok=True)

# ===== 遍历文件夹 =====
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = cv2.imread(input_path)

        if img is None:
            print(f"❌ 读取失败: {input_path}")
            continue

        img640 = letterbox(img, 640)

        cv2.imwrite(output_path, img640)

        print(f"✅ 已处理: {filename}")