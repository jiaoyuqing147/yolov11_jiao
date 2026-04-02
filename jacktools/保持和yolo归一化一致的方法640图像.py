import cv2
import numpy as np

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

img = cv2.imread("F:/DataSets/resultTT100k130val/multi_model_comparenew/TopK_vis/87590.jpg")
img640 = letterbox(img, 640)
cv2.imwrite("F:/DataSets/resultTT100k130val/multi_model_comparenew/TopK_vis/87590output.jpg", img640)
