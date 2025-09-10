# predict_full.py
import cv2

from ultralytics import YOLO
import os

# 1) 模型路径
MODEL_PATH = "runsTT100k130/yolo11_train/exp/weights/best.pt"

# 2) 预测参数（与 CLI 对齐）
PREDICT_ARGS = {
    # --- 核心必备 ---
    "source": r"E:\Datasets\ceshi",  # 图片/文件夹/视频/摄像头(0)
    "device": 0,          # 0/1/... 或 "cpu"
    "conf": 0.25,         # 置信度阈值
    "iou": 0.50,          # NMS IoU 阈值
    "save": True,         # 保存可视化结果（画框后的图）

    # --- 结果保存相关 ---
    "save_txt": True,     # 保存预测为 labels/*.txt (class x y w h [conf])
    "save_conf": True,    # 在 txt 中同时保存置信度

    # --- 可视化细节 ---
    # "show": False,        # 是否弹窗显示预测结果（GUI 环境下）
    # "save_crop": False,   # 裁剪并保存每个目标框
    "show_labels": True,  # 在图上显示类别名
    "show_conf": True,    # 在图上显示置信度
    # "line_width": None,   # 线宽, None时自动
    # "retina_masks": False,# 分割任务用(检测任务忽略)
    "boxes": True,        # 是否显示 bbox（检测任务一般保持 True）

    # --- 视频/流 ---
    # "vid_stride": 1,      # 视频帧间隔保存
    # "stream_buffer": False,# 流式缓冲(True/False)

    # --- 推理与后处理 ---
    "visualize": True,   # 可视化模型特征图
    # "augment": False,     # 推理时数据增强(TTA)
    # "agnostic_nms": False,# 类别无关的NMS
    # "classes": None,      # 仅保留指定类别，如[0,3,5]
}

def main():
    model = YOLO(MODEL_PATH)



    #3)运行预测，使用yolo自带的绘图默认模块
    #results = model.predict(**PREDICT_ARGS,name=r"E:\Datasets\ceshijieguo") #如果PREDICT_ARGS中的参数save设置为True，那就会保存到name=的路径
    '''
    结果默认保存到 runs/detect/predict 或 runs/detect/predictX
    如需自定义目录可在 predict() 里加 name="tt100k_run1"
    例如：model.predict(..., name="tt100k_run1")
    '''
    # # 4) 可选：控制台打印每个框的 像素坐标 + 类别名 + 概率
    # for r in results:
    #     names = r.names  # id -> name
    #     print(f"\nImage: {r.path}")
    #     for b in r.boxes:
    #         x1, y1, x2, y2 = b.xyxy[0].tolist()
    #         cls_id = int(b.cls[0])
    #         conf = float(b.conf[0])
    #         print(f"  [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
    #               f"{names[cls_id]} {conf:.2f}")



#---------------重新制作画板画结果-------------------------------#
    # 3) 运行预测
    results = model.predict(**PREDICT_ARGS)


    # # 自定义保存目录
    # save_dir = r"E:\Datasets\ceshijieguo"
    # os.makedirs(save_dir, exist_ok=True)
    #
    # for r in results:
    #     # 用 plot() 绘制，设置线宽、显示类别和概率
    #     im = r.plot(
    #         line_width=4,  # 加粗框线
    #         boxes=True,  # 显示框
    #         labels=True,  # 显示类别名
    #         conf=True  # 显示置信度
    #     )
    #
    #     # 保存结果图像
    #     save_path = os.path.join(save_dir, os.path.basename(r.path))
    #     cv2.imwrite(save_path, im)
    #
    #     # 控制台打印检测结果
    #     print(f"\nImage: {r.path}")
    #     for b in r.boxes:
    #         x1, y1, x2, y2 = b.xyxy[0].tolist()
    #         cls_id = int(b.cls[0])
    #         conf = float(b.conf[0])
    #         print(f"  [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] "
    #               f"{r.names[cls_id]} {conf:.2f}")
    #




if __name__ == "__main__":
    main()
