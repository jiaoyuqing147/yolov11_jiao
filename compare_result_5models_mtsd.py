import os
import csv
import shutil
from pathlib import Path
from typing import List, Tuple, Dict
from PIL import Image
from tqdm import tqdm

# ================== 配置 ==================
IMAGES_DIR = Path(r"E:\DataSets\MTSD\yolo54\images\train")
GT_LABELS  = Path(r"E:\DataSets\MTSD\yolo54\labels\train")

# 多模型预测目录
# 多模型预测目录
MODELS = {
    "A": Path(r"E:\DataSets\MTSDresult\yolo11-FASFFHead_P234_RCSOSA_wiou_bce_distillation"),
    "B": Path(r"E:\DataSets\MTSDresult\yolo11-FASFFHead_P234_RCSOSA_wiou_bce_train"),
    "C": Path(r"E:\DataSets\MTSDresult\yolo11-FASFFHead_P234_RCSOSA_ciou_bce_train"),
    "D": Path(r"E:\DataSets\MTSDresult\yolo11-FASFFHead_P234_train"),
    "E": Path(r"E:\DataSets\MTSDresult\yolo11_train"),
}

ORDER  = ["A","B","C","D","E"]  # 模型顺序
METRIC = "f1"                   # 或 "ap50"
STRICT = False                   # True: 严格 >；False: 允许 ≥
EPS    = 1e-6                   # 容差

IOU_THR      = 0.50
CONF_THR_PRF = 0.25
IMG_EXTS     = (".jpg", ".jpeg", ".png", ".bmp")

OUT_DIR   = Path(r"E:\DataSets\MTSDresult\multi_model_compare")
OUT_CSV   = OUT_DIR / f"per_image_{METRIC}_order.csv"
COPY_TOPK = 3  # Top-K 差值最大（符合排序）拷贝
COPY_BOTK = 3    # Bottom-K 差值最小（排序最弱）拷贝
COPY_DIR_TOP = OUT_DIR / "TopK"
COPY_DIR_BOT = OUT_DIR / "BottomK"

# ================== 工具函数 ==================
def yolo_to_xyxy(norm_box, w, h):
    cx, cy, bw, bh = norm_box
    return (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1,bx1), max(ay1,by1), min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)
    return inter / (area_a + area_b - inter)

def load_gt_txt(txt_path):
    out=[]
    if txt_path.exists():
        for line in open(txt_path, "r", encoding="utf-8"):
            p=line.strip().split()
            if len(p)>=5:
                out.append((int(float(p[0])), *map(float, p[1:5])))
    return out

def load_pred_txt(txt_path):
    out=[]
    if txt_path.exists():
        for line in open(txt_path, "r", encoding="utf-8"):
            p=line.strip().split()
            if len(p)>=6:
                out.append((int(float(p[0])), *map(float, p[1:6])))
    return out

def match_by_iou(gts_xyxy, preds_xyxy, thr):
    order = sorted(range(len(preds_xyxy)), key=lambda i: preds_xyxy[i][2], reverse=True)
    gt_used = [False]*len(gts_xyxy)
    tp=fp=0
    for pi in order:
        p_cls, p_box, _ = preds_xyxy[pi]
        best_iou,best_gt=0,-1
        for gi,(g_cls,g_box) in enumerate(gts_xyxy):
            if gt_used[gi] or g_cls!=p_cls: continue
            iou=iou_xyxy(p_box,g_box)
            if iou>best_iou:
                best_iou,best_gt=iou,gi
        if best_iou>=thr and best_gt>=0:
            gt_used[best_gt]=True
            tp+=1
        else:
            fp+=1
    fn = sum(1 for u in gt_used if not u)
    return tp,fp,fn

def prf_from_match(tp,fp,fn):
    P=tp/(tp+fp) if tp+fp>0 else 0
    R=tp/(tp+fn) if tp+fn>0 else 0
    F1=2*P*R/(P+R) if P+R>0 else 0
    return P,R,F1

def ap50_from_preds(gts_xyxy,preds_xyxy,thr):
    preds=sorted(preds_xyxy,key=lambda x:x[2],reverse=True)
    gt_flags=[False]*len(gts_xyxy)
    tps=fps=0;precisions=[];recalls=[]
    total_gt=len(gts_xyxy)
    for p_cls,p_box,_ in preds:
        best_iou,best_gt=0,-1
        for gi,(g_cls,g_box) in enumerate(gts_xyxy):
            if gt_flags[gi] or g_cls!=p_cls: continue
            iou=iou_xyxy(p_box,g_box)
            if iou>best_iou: best_iou,best_gt=iou,gi
        if best_iou>=thr and best_gt>=0:
            gt_flags[best_gt]=True;tps+=1
        else:
            fps+=1
        precisions.append(tps/(tps+fps) if tps+fps>0 else 0)
        recalls.append(tps/total_gt if total_gt>0 else 0)
    ap=0;prev_r,prev_p=0,1
    for r,p in sorted(zip(recalls,precisions)):
        ap+=(r-prev_r)*(p+prev_p)/2
        prev_r,prev_p=r,p
    return ap

def evaluate_one_image(img_path, gt_dir, pred_dir):
    with Image.open(img_path) as im: W,H=im.size
    gts = [(c,yolo_to_xyxy((cx,cy,bw,bh),W,H)) for c,cx,cy,bw,bh in load_gt_txt(gt_dir/f"{img_path.stem}.txt")]
    preds_all=[(c,yolo_to_xyxy((cx,cy,bw,bh),W,H),s) for c,cx,cy,bw,bh,s in load_pred_txt(pred_dir/f"{img_path.stem}.txt")]
    preds_keep=[p for p in preds_all if p[2]>=CONF_THR_PRF]
    tp,fp,fn=match_by_iou(gts,preds_keep,IOU_THR)
    P,R,F1=prf_from_match(tp,fp,fn)
    return {"n_gt":len(gts),"n_pred":len(preds_keep),"precision":P,"recall":R,"f1":F1,"ap50":ap50_from_preds(gts,preds_all,IOU_THR)}

def check_order(values, strict, eps=0.0):
    for i in range(len(values)-1):
        if strict:
            if not (values[i]>values[i+1]+eps): return False
        else:
            if not (values[i]+eps>=values[i+1]): return False
    return True

# ================== 主程序 ==================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_list = sorted([p for ext in IMG_EXTS for p in IMAGES_DIR.rglob(f"*{ext}")])
    rows=[]
    for img_path in tqdm(img_list, desc="Evaluating", ncols=100):
        per_model={name:evaluate_one_image(img_path,GT_LABELS,pred_dir) for name,pred_dir in MODELS.items()}
        metric_values=[per_model[name][METRIC] for name in ORDER]
        ok=check_order(metric_values,STRICT,EPS)
        row={"image":img_path.name}
        for name in MODELS:
            m=per_model[name]
            row.update({f"{name}_F1":m["f1"],f"{name}_AP50":m["ap50"]})
        row["match"]=int(ok)
        rows.append(row)

    # 保存 CSV
    with open(OUT_CSV,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(row.keys())
        for r in rows: w.writerow(r.values())
    print(f"[OK] CSV saved -> {OUT_CSV}")

    # 拷贝 top/bottom
    rows_match=[r for r in rows if r["match"]==1]
    rows_match.sort(key=lambda r: sum(r[f"{name}_F1"] for name in ORDER), reverse=True)
    COPY_DIR_TOP.mkdir(parents=True, exist_ok=True)
    for r in rows_match[:COPY_TOPK]:
        shutil.copy(IMAGES_DIR/r["image"],COPY_DIR_TOP/r["image"])
    COPY_DIR_BOT.mkdir(parents=True, exist_ok=True)
    for r in rows_match[-COPY_BOTK:]:
        shutil.copy(IMAGES_DIR/r["image"],COPY_DIR_BOT/r["image"])
    print(f"[OK] Copied top {COPY_TOPK} and bottom {COPY_BOTK} images.")

if __name__=="__main__":
    main()
