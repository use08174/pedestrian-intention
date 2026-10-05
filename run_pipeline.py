"""Local port of the notebook's main pipeline (cells 58 -> 60 -> 61 -> 73 -> 77).

Logic is kept as in the notebook; only Kaggle paths are replaced with local ones.
Added: evaluation on the test split (the notebook only reports the val split).
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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data')
BASE_PATH = os.path.join(DATA, 'vehic-ped-intuition')
XML_PATH = os.path.join(DATA, 'attributes-label', 'annotations_attributes')
MODEL_PATH = os.path.join(DATA, 'first-phase-model', 'weights', 'best.pt')
OUT = os.path.join(ROOT, 'outputs')
OUTPUT_DIR = os.path.join(OUT, 'perception_results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQ_LEN = 10
FEATURES = ['x', 'y', 'vel_x', 'vel_y', 'delta_area', 'aspect_ratio']
TARGET = 'label'

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ---------------- Phase 2: tracking + feature extraction (cell 58) ----------------
def enhance_image(img):
    if img is None:
        return None
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def process_split(yolo, split_name):
    print(f"\nPROCESSING SPLIT: {split_name.upper()}...", flush=True)
    img_dir = os.path.join(BASE_PATH, 'images', split_name)
    all_files = os.listdir(img_dir)
    video_ids = sorted(list(set([f.split('_')[1] for f in all_files if '_' in f])))
    split_rows = []

    for n, v_id in enumerate(video_ids):
        xml_file = os.path.join(XML_PATH, f"video_{v_id}_attributes.xml")
        if not os.path.exists(xml_file):
            continue
        peds_meta = {}
        tree = ET.parse(xml_file)
        for p in tree.getroot().findall('pedestrian'):
            peds_meta[p.get('id')] = {'dp': int(p.get('decision_point')),
                                      'crossing': int(p.get('crossing'))}

        v_frames = sorted([os.path.join(img_dir, f) for f in all_files if f"video_{v_id}" in f])
        processed_tracks = defaultdict(list)
        id_map = {}
        assigned_xml_ids = set()

        for fidx, path in enumerate(v_frames[:100]):
            img = cv2.imread(path)
            img_enh = enhance_image(img)
            results = yolo.track(img_enh, persist=True, conf=0.28, classes=[0], verbose=False)[0]
            if results.boxes.id is not None:
                t_boxes = results.boxes.xyxy.cpu().numpy()
                t_ids = results.boxes.id.int().cpu().numpy()
                for tb, tid in zip(t_boxes, t_ids):
                    if tid not in id_map:
                        for xid in peds_meta.keys():
                            if xid not in assigned_xml_ids:
                                id_map[tid] = xid
                                assigned_xml_ids.add(xid)
                                break
                    if tid in id_map:
                        current_xid = id_map[tid]
                        processed_tracks[current_xid].append({
                            'frame': fidx, 'bbox': tb,
                            'before_dp': fidx <= peds_meta[current_xid]['dp']})

        for xid, frames_list in processed_tracks.items():
            if len(frames_list) < 5:
                continue
            for i in range(len(frames_list)):
                curr = frames_list[i]
                b = curr['bbox']
                cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]
                vel_x, vel_y, d_area = (0, 0, 1)
                if i > 0:
                    p_b = frames_list[i - 1]['bbox']
                    vel_x = cx - (p_b[0] + p_b[2]) / 2
                    vel_y = cy - (p_b[1] + p_b[3]) / 2
                    d_area = (w * h) / ((p_b[2] - p_b[0]) * (p_b[3] - p_b[1]) + 1e-6)
                split_rows.append({
                    'video_id': v_id, 'ped_id': xid, 'frame': curr['frame'],
                    'x': cx, 'y': cy, 'vel_x': vel_x, 'vel_y': vel_y,
                    'delta_area': d_area, 'aspect_ratio': w / h,
                    'before_dp': int(curr['before_dp']),
                    'label': peds_meta[xid]['crossing']})
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(video_ids)} videos", flush=True)

    df = pd.DataFrame(split_rows)
    out = os.path.join(OUT, f"master_{split_name}_dataset.csv")
    df.to_csv(out, index=False)
    print(f"Saved: {out} ({len(df)} rows)", flush=True)


# ---------------- Phase 3: sequences (cells 60 + 61) ----------------
scaler = StandardScaler()


def create_lstm_sequences(csv_path, seq_len=10, is_training=False):
    df = pd.read_csv(csv_path)
    df = df[df[TARGET].isin([0, 1])]
    df[FEATURES] = df[FEATURES].fillna(0)
    feature_values = df[FEATURES].values
    scaled_values = scaler.fit_transform(feature_values) if is_training else scaler.transform(feature_values)
    df[FEATURES] = scaled_values
    sequences, labels = [], []
    for (v_id, p_id), group in df.groupby(['video_id', 'ped_id']):
        if len(group) >= seq_len:
            group = group.sort_values('frame')
            feature_data = group[FEATURES].values
            target_value = group[TARGET].mode()[0]
            for i in range(len(group) - seq_len + 1):
                sequences.append(feature_data[i:i + seq_len])
                labels.append(target_value)
    return np.array(sequences), np.array(labels)


# ---------------- Model (cell 88, with output_size as used in cells 70/73) ----------------
class IntentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])


def evaluate(model, loader, name, timestamp):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for bx, by in loader:
            preds = (torch.sigmoid(model(bx.to(device))) > 0.5).float()
            all_preds.extend(preds.cpu().numpy().ravel())
            all_labels.extend(by.numpy().ravel())
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['Stay', 'Cross'],
                yticklabels=['Stay', 'Cross'], annot_kws={"size": 16, "weight": "bold"})
    plt.xlabel('AI Prediction')
    plt.ylabel('Actual Reality (Label)')
    plt.title(f'Confusion Matrix ({name})')
    plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}_{timestamp}.png"), dpi=150)
    plt.close()
    report = classification_report(all_labels, all_preds, labels=[0, 1],
                                   target_names=['Stay', 'Cross'], zero_division=0)
    with open(os.path.join(OUTPUT_DIR, f"classification_report_{name}_{timestamp}.txt"), 'w') as f:
        f.write(report)
    print(f"\n--- Classification Report ({name}) ---\n{report}\nConfusion matrix [[TN FP][FN TP]]:\n{cm}", flush=True)


def main():
    if '--skip-extract' not in sys.argv:
        yolo = YOLO(MODEL_PATH)
        for split in ['train', 'val', 'test']:
            yolo = YOLO(MODEL_PATH)  # fresh tracker state per split
            process_split(yolo, split)

    # Oversampling (cell 60)
    df_raw = pd.read_csv(os.path.join(OUT, 'master_train_dataset.csv'))
    df_raw = df_raw[df_raw[TARGET].isin([0, 1])]
    df_majority = df_raw[df_raw[TARGET] == 1]
    df_minority = df_raw[df_raw[TARGET] == 0]
    print(f"\nTrain rows before balancing: cross={len(df_majority)} stay={len(df_minority)}")
    df_minority_up = df_minority.sample(len(df_majority), replace=True, random_state=42)
    pd.concat([df_majority, df_minority_up]).to_csv(os.path.join(OUT, 'balanced_train_dataset.csv'), index=False)

    X_train, y_train = create_lstm_sequences(os.path.join(OUT, 'balanced_train_dataset.csv'), SEQ_LEN, True)
    X_val, y_val = create_lstm_sequences(os.path.join(OUT, 'master_val_dataset.csv'), SEQ_LEN)
    X_test, y_test = create_lstm_sequences(os.path.join(OUT, 'master_test_dataset.csv'), SEQ_LEN)
    for n, y in [('train', y_train), ('val', y_val), ('test', y_test)]:
        print(f"{n} sequences: {len(y)} (stay={int((y == 0).sum())}, cross={int((y == 1).sum())})")

    mk = lambda X, y, sh: torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(y).reshape(-1, 1)), batch_size=64, shuffle=sh)
    train_loader, val_loader, test_loader = mk(X_train, y_train, True), mk(X_val, y_val, False), mk(X_test, y_test, False)

    # Training loop (cell 73)
    model = IntentionLSTM(6, 64, 2, 1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3)
    PATIENCE, best_val_loss, counter = 15, float('inf'), 0
    best_path = os.path.join(OUT, 'best_pedestrian_model.pth')
    hist = defaultdict(list)
    for epoch in range(40):
        model.train()
        tl, tc = 0.0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
    evaluate(model, val_loader, 'val', ts)
    evaluate(model, test_loader, 'test', ts)


if __name__ == '__main__':
    main()
