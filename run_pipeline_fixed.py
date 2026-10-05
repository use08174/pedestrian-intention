"""Fixed version of run_pipeline.py.

Fixes vs. the notebook logic:
  1. Label assignment: YOLO detections are matched to JAAD ground-truth pedestrian boxes
     per frame by IoU (Hungarian, IoU >= 0.5) instead of handing out XML ids in order.
  2. Oversampling is done on sequences (after windowing), not on individual frame rows.
  3. Original frame numbers are parsed from filenames (frames are sampled every 5), so
     before_dp is compared on the same scale as decision_point; tracks are split at gaps.
  4. The scaler is fit on the un-oversampled train rows.
  5. Reports macro-F1 / balanced accuracy and the always-"Cross" baseline.

Usage: python run_pipeline_fixed.py [--skip-extract] [--before-dp]
  --before-dp  only use frames up to the pedestrian's decision point (intention, not action)
"""
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

from run_pipeline import BASE_PATH, MODEL_PATH, XML_PATH, ROOT, IntentionLSTM, enhance_image

JAAD_ANN = os.path.join(ROOT, 'data', 'JAAD', 'annotations')
BEFORE_DP = '--before-dp' in sys.argv
OUT = os.path.join(ROOT, 'outputs_fixed')
OUTPUT_DIR = os.path.join(OUT, 'perception_results_before_dp' if BEFORE_DP else 'perception_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 640          # frames were stretched from 1920x1080 to 640x640
FRAME_STRIDE = 5        # frames are sampled every 5 original frames
IOU_THR = 0.5
SEQ_LEN = 10
FEATURES = ['x', 'y', 'vel_x', 'vel_y', 'delta_area', 'aspect_ratio']
TARGET = 'label'

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_gt_boxes(v_id):
    """{frame: [(ped_id, [x1, y1, x2, y2] in 640x640 coords)]} for all annotated pedestrians."""
    root = ET.parse(os.path.join(JAAD_ANN, f"video_{v_id}.xml")).getroot()
    W = float(root.find('.//original_size/width').text)
    H = float(root.find('.//original_size/height').text)
    sx, sy = IMG_SIZE / W, IMG_SIZE / H
    gt = defaultdict(list)
    for t in root.findall('track'):
        if t.get('label') not in ('pedestrian', 'ped'):
            continue
        for b in t.findall('box'):
            if b.get('outside') != '0':
                continue
            pid = next((a.text for a in b.findall('attribute') if a.get('name') == 'id'), None)
            box = [float(b.get('xtl')) * sx, float(b.get('ytl')) * sy,
                   float(b.get('xbr')) * sx, float(b.get('ybr')) * sy]
            gt[int(b.get('frame'))].append((pid, box))
    return gt


def iou_matrix(a, b):
    a, b = np.asarray(a)[:, None, :], np.asarray(b)[None, :, :]
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def process_split(yolo, split_name):
    print(f"\nPROCESSING SPLIT: {split_name.upper()}...", flush=True)
    img_dir = os.path.join(BASE_PATH, 'images', split_name)
    all_files = os.listdir(img_dir)
    video_ids = sorted(set(f.split('_')[1] for f in all_files if '_' in f))
    rows, n_det, n_matched = [], 0, 0

    for v_id in video_ids:
        xml_file = os.path.join(XML_PATH, f"video_{v_id}_attributes.xml")
        if not os.path.exists(xml_file):
            continue
        peds_meta = {p.get('id'): {'dp': int(p.get('decision_point')), 'crossing': int(p.get('crossing'))}
                     for p in ET.parse(xml_file).getroot().findall('pedestrian')}
        gt = load_gt_boxes(v_id)

        v_frames = sorted(f for f in all_files if f.startswith(f"video_{v_id}_"))
        tracks = defaultdict(list)
        for fname in v_frames:
            frame_no = int(fname.split('_')[2][1:5])
            img_enh = enhance_image(cv2.imread(os.path.join(img_dir, fname)))
            det = yolo.predict(img_enh, conf=0.28, classes=[0], verbose=False)[0].boxes.xyxy.cpu().numpy()
            n_det += len(det)
            gts = gt.get(frame_no, [])
            if len(det) == 0 or len(gts) == 0:
                continue
            iou = iou_matrix(det, [g[1] for g in gts])
            for di, gi in zip(*linear_sum_assignment(-iou)):
                if iou[di, gi] < IOU_THR:
                    continue
                n_matched += 1
                pid = gts[gi][0]
                if pid in peds_meta:  # only behaviour-annotated pedestrians carry a crossing label
                    tracks[pid].append({'frame': frame_no, 'bbox': det[di]})

        for pid, fl in tracks.items():
            fl.sort(key=lambda d: d['frame'])
            streak = 0
            for i, curr in enumerate(fl):
                b = curr['bbox']
                cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]
                vel_x, vel_y, d_area = 0, 0, 1
                if i > 0 and curr['frame'] - fl[i - 1]['frame'] == FRAME_STRIDE:
                    p = fl[i - 1]['bbox']
                    vel_x, vel_y = cx - (p[0] + p[2]) / 2, cy - (p[1] + p[3]) / 2
                    d_area = (w * h) / ((p[2] - p[0]) * (p[3] - p[1]) + 1e-6)
                elif i > 0:
                    streak += 1  # gap in detections -> start a new streak
                rows.append({'video_id': v_id, 'ped_id': pid, 'streak': streak, 'frame': curr['frame'],
                             'x': cx, 'y': cy, 'vel_x': vel_x, 'vel_y': vel_y,
                             'delta_area': d_area, 'aspect_ratio': w / h,
                             'before_dp': int(curr['frame'] <= peds_meta[pid]['dp']),
                             'label': peds_meta[pid]['crossing']})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"master_{split_name}_dataset.csv"), index=False)
    print(f"Saved master_{split_name}_dataset.csv ({len(df)} rows) | detections={n_det}, "
          f"IoU-matched to GT={n_matched} ({n_matched / max(n_det, 1):.0%})", flush=True)


def load_rows(split):
    df = pd.read_csv(os.path.join(OUT, f"master_{split}_dataset.csv"), dtype={'video_id': str})
    df = df[df[TARGET].isin([0, 1])]
    if BEFORE_DP:
        df = df[df['before_dp'] == 1]
    return df


def make_sequences(df, scaler):
    df = df.copy()
    df[FEATURES] = scaler.transform(df[FEATURES].fillna(0).values)
    seqs, labels = [], []
    for _, g in df.groupby(['video_id', 'ped_id', 'streak']):
        if len(g) < SEQ_LEN:
            continue
        g = g.sort_values('frame')
        feat, y = g[FEATURES].values, int(g[TARGET].iloc[0])
        for i in range(len(g) - SEQ_LEN + 1):
            seqs.append(feat[i:i + SEQ_LEN])
            labels.append(y)
    return np.array(seqs, dtype=np.float32).reshape(-1, SEQ_LEN, len(FEATURES)), np.array(labels)


def evaluate(model, X, y, name, ts):
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(X).to(device))).cpu().numpy().ravel()
    pred = (prob > 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Stay', 'Cross'],
                yticklabels=['Stay', 'Cross'], annot_kws={"size": 16, "weight": "bold"})
    plt.xlabel('AI Prediction'); plt.ylabel('Actual Reality (Label)'); plt.title(f'Confusion Matrix ({name})')
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}_{ts}.png"), dpi=150); plt.close()
    report = classification_report(y, pred, labels=[0, 1], target_names=['Stay', 'Cross'], zero_division=0)
    summary = (f"accuracy={np.mean(pred == y):.3f}  balanced_acc={balanced_accuracy_score(y, pred):.3f}  "
               f"macro_f1={f1_score(y, pred, average='macro'):.3f}  |  always-Cross baseline: "
               f"accuracy={np.mean(y == 1):.3f}  balanced_acc=0.500  "
               f"macro_f1={f1_score(y, np.ones_like(y), average='macro', zero_division=0):.3f}")
    with open(os.path.join(OUTPUT_DIR, f"classification_report_{name}_{ts}.txt"), 'w') as f:
        f.write(report + '\n' + summary + '\n')
    print(f"\n--- {name} ---\n{report}{summary}\nConfusion matrix [[TN FP][FN TP]]:\n{cm}", flush=True)


def main():
    if '--skip-extract' not in sys.argv:
        yolo = YOLO(MODEL_PATH)
        for split in ['train', 'val', 'test']:
            process_split(yolo, split)

    tr, va, te = load_rows('train'), load_rows('val'), load_rows('test')
    for n, d in [('train', tr), ('val', va), ('test', te)]:
        print(f"{n} rows: {len(d)} | pedestrians stay={d[d.label == 0].ped_id.nunique()} "
              f"cross={d[d.label == 1].ped_id.nunique()}")
    scaler = StandardScaler().fit(tr[FEATURES].fillna(0).values)
    X_train, y_train = make_sequences(tr, scaler)
    X_val, y_val = make_sequences(va, scaler)
    X_test, y_test = make_sequences(te, scaler)

    # Sequence-level oversampling of the minority class (train only)
    cls, cnt = np.unique(y_train, return_counts=True)
    minority = cls[np.argmin(cnt)]
    idx_min = np.where(y_train == minority)[0]
    extra = np.random.choice(idx_min, cnt.max() - cnt.min(), replace=True)
    X_train, y_train = np.concatenate([X_train, X_train[extra]]), np.concatenate([y_train, y_train[extra]])
    for n, y in [('train (balanced)', y_train), ('val', y_val), ('test', y_test)]:
        print(f"{n} sequences: {len(y)} (stay={int((y == 0).sum())}, cross={int((y == 1).sum())})")

    mk = lambda X, y, sh: torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)),
        batch_size=64, shuffle=sh)
    train_loader, val_loader = mk(X_train, y_train, True), mk(X_val, y_val, False)

    # Training loop (same hyper-parameters as notebook cell 73)
    model = IntentionLSTM(6, 64, 2, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    PATIENCE, best_val_loss, counter = 15, float('inf'), 0
    best_path = os.path.join(OUTPUT_DIR, 'best_pedestrian_model.pth')
    hist = defaultdict(list)
    for epoch in range(40):
        model.train()
        tl, tc = 0.0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tl += loss.item()
            tc += ((torch.sigmoid(logits) > 0.5).float() == by).sum().item()
        model.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                vlog = model(bx)
                vl += criterion(vlog, by).item()
                vc += ((torch.sigmoid(vlog) > 0.5).float() == by).sum().item()
        etl, evl = tl / len(train_loader), vl / len(val_loader)
        eta, eva = tc / len(train_loader.dataset) * 100, vc / len(val_loader.dataset) * 100
        for k, v in zip(['tl', 'vl', 'ta', 'va'], [etl, evl, eta, eva]):
            hist[k].append(v)
        msg = f"Epoch [{epoch + 1}] | Train Loss {etl:.4f} Acc {eta:.2f}% | Val Loss {evl:.4f} Acc {eva:.2f}%"
        if evl < best_val_loss:
            best_val_loss, counter = evl, 0
            torch.save(model.state_dict(), best_path)
            msg += "  * best"
        else:
            counter += 1
        print(msg, flush=True)
        if counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ep = range(1, len(hist['tl']) + 1)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1); plt.plot(ep, hist['tl'], 'o-', label='Train Loss'); plt.plot(ep, hist['vl'], 'o-', label='Val Loss'); plt.legend(); plt.grid(alpha=.5)
    plt.subplot(1, 2, 2); plt.plot(ep, hist['ta'], 's--', label='Train Acc'); plt.plot(ep, hist['va'], 's--', label='Val Acc'); plt.legend(); plt.grid(alpha=.5)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, f"training_metrics_{ts}.png"), dpi=150); plt.close()

    model.load_state_dict(torch.load(best_path))
    model.eval()
    evaluate(model, X_val, y_val, 'val', ts)
    evaluate(model, X_test, y_test, 'test', ts)


if __name__ == '__main__':
    main()
