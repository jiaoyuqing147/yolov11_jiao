# 将txt格式的目标识别标签文件转化为在原图基础上的识别结果图像，其中有类别ID和置信度
# -*- coding: utf-8 -*-
import os
import cv2


class Colors:
    # Ultralytics color palette
    def __init__(self):
        hexs = (
            'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231',
            '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
            '2C99A8', '00C2FF', '344593', '6473FF', '0018EC',
            '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7'
        )
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()

# ================= 路径配置 =================
path = r"C:\Users\Administrator\Desktop\320\labels"       # txt标签目录
path2 = r"C:\Users\Administrator\Desktop\320\images"      # 原图目录
save_path = r"C:\Users\Administrator\Desktop\320\images_see"  # 保存目录

os.makedirs(save_path, exist_ok=True)

# 支持的图片后缀
img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF"}

file_list = os.listdir(path2)

for imgname in file_list:
    img_path = os.path.join(path2, imgname)

    # 跳过非图片
    if not os.path.isfile(img_path):
        continue
    if os.path.splitext(imgname)[1] not in img_exts:
        continue

    # 用通用方式找对应txt
    basename = os.path.splitext(imgname)[0]
    labelsname = basename + ".txt"
    label_path = os.path.join(path, labelsname)

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ 图片读取失败: {img_path}")
        continue

    height_real, width_real = img.shape[:2]

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            test = f.readlines()

        draw_count = 0
        print(label_path)

        for line in test:
            line = line.strip()
            if not line:
                continue

            a = line.split()

            # 你的预测文件格式应为：cls x y w h conf
            if len(a) < 6:
                continue

            # 类别过滤
            if a[0] == '811':
                continue

            cls_id = int(float(a[0]))
            x = float(a[1])
            y = float(a[2])
            w = float(a[3])
            h = float(a[4])
            conf = float(a[5])

            # 置信度过滤
            if conf < 0.5:
                continue

            # YOLO归一化坐标 -> 像素坐标
            x_center = width_real * x
            y_center = height_real * y
            box_w = width_real * w
            box_h = height_real * h

            x1 = int(x_center - box_w / 2.0)
            y1 = int(y_center - box_h / 2.0)
            x2 = int(x_center + box_w / 2.0)
            y2 = int(y_center + box_h / 2.0)

            # 裁剪到图像范围内
            x1 = max(0, min(x1, width_real - 1))
            y1 = max(0, min(y1, height_real - 1))
            x2 = max(0, min(x2, width_real - 1))
            y2 = max(0, min(y2, height_real - 1))

            # 无效框跳过
            if x2 <= x1 or y2 <= y1:
                continue

            color = colors(cls_id, bgr=True)

            # 画框
            cv2.rectangle(
                img,
                (x1, y1),
                (x2, y2),
                color,
                thickness=max(round(sum(img.shape) / 2 * 0.001), 2),
                lineType=cv2.LINE_AA
            )

            # 标签：类别ID + 置信度
            label = f"{cls_id} {conf:.2f}"

            # 文字背景框大小
            font_scale = max(sum(img.shape) / 2 * 0.0008, 0.4)
            font_thickness = max(round(sum(img.shape) / 2 * 0.002), 1)
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

            outside = y1 - th - baseline >= 3
            if outside:
                txt_bg_tl = (x1, y1 - th - baseline - 2)
                txt_bg_br = (x1 + tw, y1)
                txt_org = (x1, y1 - baseline - 2)
            else:
                txt_bg_tl = (x1, y1)
                txt_bg_br = (x1 + tw, y1 + th + baseline + 2)
                txt_org = (x1, y1 + th)

            # 画文字背景
            cv2.rectangle(img, txt_bg_tl, txt_bg_br, color, -1, cv2.LINE_AA)

            # 写文字
            cv2.putText(
                img,
                label,
                txt_org,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )

            draw_count += 1

    except FileNotFoundError:
        print(f"❌ 文件未找到: {label_path}")

    out_path = os.path.join(save_path, imgname)
    cv2.imwrite(out_path, img)
    print(f"✅ 已保存: {out_path}，画了 {draw_count if 'draw_count' in locals() else 0} 个框")

print("🎉 所有图片可视化完成！🚀")