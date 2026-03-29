# 将txt格式的目标识别标签文件转化为在原图基础上的识别结果图像，其中有图像序号，类别
# -*- coding: utf-8 -*-
import os, re
import cv2
import numpy as np
from category import get_category
from PIL import ImageFont, ImageDraw, Image


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

path = "E:\DataSets\forpaper\ceshiTT100Kresult_yolo11_FASFFHead_P234"  # labels地址
path2 = "E:\DataSets\forpaper\ceshiTT100Kresult_yolo11_FASFFHead_P234"  # 原图文件地址\
save_path = "E:\DataSets\forpaper\ceshiTT100Kresult_yolo11_FASFFHead_P234"  # 保存文件的地址

file = os.listdir(path2)
for imgname in file:
    labelsname = re.sub('.JPEG', ".txt", imgname)
    img = cv2.imread(path2 + '/' + imgname)
    size = img.shape
    height_real = size[0]
    width_real = size[1]
    try:
        with open(path + "/" + labelsname, "r") as f:
            test = f.readlines()
            num = 0
            print(path + "/" + labelsname)
            for i in test:
                box = []
                num += 1
                i = re.sub('\n', "", i)
                a = i.split(' ')
                if a[0] != '811':
                    if float(a[5]) >= 0.5:
                        box.append(a[1])
                        box.append(a[2])
                        box.append(a[3])
                        box.append(a[4])
                        # label=str(num)+' '+get_category(int(a[0]))
                        label = get_category(int(a[0])) + ' ' + a[5][0:4]

                        x_center = int(width_real * float(box[0]))  # aa[1]左上点的x坐标
                        y_center = int(height_real * float(box[1]))  # aa[2]左上点的y坐标
                        width = int(width_real * float(box[2]))  # aa[3]图片width
                        height = int(height_real * float(box[3]))  # aa[4]图片height
                        x_center = int(x_center - width / 2.0)
                        y_center = int(y_center - height / 2.0)
                        cv2.rectangle(img, (x_center, y_center), (x_center + width, y_center + height),
                                      colors(int(a[0])),
                                      thickness=max(round(sum(img.shape) / 2 * 0.001), 2),
                                      lineType=cv2.LINE_AA)  # 框细胞的框框
                        tf = max(max(round(sum(img.shape) / 2 * 0.003), 2) - 1, 1)  # font thickness
                        # text width, height
                        font = ImageFont.truetype('/home/zwc/cell/transdata/Arial Unicode.ttf',
                                                  max(round(sum(img.shape) / 2 * 0.015), 175))
                        w, h = font.getsize(label)
                        outside = (x_center, y_center)[1] - h >= 3  # 指标签是否在图片之外即细胞在图片边缘
                        p2 = (x_center, y_center)[0] + w, (x_center, y_center)[1] - h if outside else \
                        (x_center, y_center)[1] + h
                        cv2.rectangle(img, (x_center, y_center), p2, colors(int(a[0])), -
                        1, cv2.LINE_AA)  # filled框字符的框框
                        # 加中文给图片
                        img_pil = Image.fromarray(img)

                        ImageDraw.Draw(img_pil).text(
                            (x_center, y_center - h if outside else y_center), label, fill=(255, 255, 255),
                            font=font)  # 字符的颜色
                        img = np.array(img_pil)

                f.close()
    except FileNotFoundError:
        print("文件未找到")
    cv2.imwrite(save_path + '//' + imgname, img)