"""Top-down pose extraction: JAAD-finetuned YOLO detector -> YOLO26-pose on each person crop.

Off-the-shelf pose models find only ~50% of the labelled JAAD pedestrians on these frames (val subset,
IoU>=0.5), versus ~73% for the finetuned detector (data/first-phase-model), so boxes come from the
finetuned detector and YOLO26x-pose is run on an un-stretched, padded crop of every matched box.

Output: outputs_pose/pose_{split}.pkl, one row per (video, pedestrian, frame) with the detected box
(640x640 coords), 17 COCO keypoints (x, y, conf; conf=0 if pose found nothing), label and decision point.

Usage: python pose_extract.py [--model yolo26x-pose.pt]
"""
import os
import sys
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from run_pipeline import BASE_PATH, MODEL_PATH, XML_PATH, ROOT, enhance_image
from run_pipeline_fixed import IOU_THR, iou_matrix, load_gt_boxes

POSE_MODEL = sys.argv[sys.argv.index('--model') + 1] if '--model' in sys.argv else 'yolo26x-pose.pt'
OUT = os.path.join(ROOT, 'outputs_pose')
os.makedirs(OUT, exist_ok=True)
SX = 1920 / 640           # un-stretch: frames were resized 1920x1080 -> 640x640
SY = 1080 / 640
PAD = 0.25                # crop margin, fraction of box size
CROP_H = 384              # crops are resized to this height before pose


def crop_pose(pose, full, boxes):
    """Run pose on padded crops of `boxes` (640x640 coords). Returns kx, ky (640 coords), kc: (n, 17)."""
    crops, metas = [], []
    for x1, y1, x2, y2 in boxes:
        X1, Y1, X2, Y2 = x1 * SX, y1 * SY, x2 * SX, y2 * SY
        pw, ph = (X2 - X1) * PAD, (Y2 - Y1) * PAD
        cx1, cy1 = int(max(X1 - pw, 0)), int(max(Y1 - ph, 0))
        cx2, cy2 = int(min(X2 + pw, full.shape[1])), int(min(Y2 + ph, full.shape[0]))
        c = full[cy1:cy2, cx1:cx2]
        s = CROP_H / max(c.shape[0], 1)
        crops.append(cv2.resize(c, (max(int(c.shape[1] * s), 8), CROP_H)))
        metas.append((cx1, cy1, s, (X1 - cx1) * s, (Y1 - cy1) * s, (X2 - cx1) * s, (Y2 - cy1) * s))
    kx, ky, kc = np.zeros((len(boxes), 17)), np.zeros((len(boxes), 17)), np.zeros((len(boxes), 17))
    for i, (res, (cx1, cy1, s, *tb)) in enumerate(zip(pose.predict(crops, imgsz=CROP_H, conf=0.1, verbose=False), metas)):
        if len(res.boxes) == 0:
            continue
        # pick the pose detection that best overlaps the target box inside the crop
        j = int(np.argmax(iou_matrix(res.boxes.xyxy.cpu().numpy(), [tb])[:, 0]))
        xy, cf = res.keypoints.xy[j].cpu().numpy(), res.keypoints.conf[j].cpu().numpy()
        kx[i] = (xy[:, 0] / s + cx1) / SX
        ky[i] = (xy[:, 1] / s + cy1) / SY
        kc[i] = cf
    return kx, ky, kc


def process_split(det, pose, split):
    img_dir = os.path.join(BASE_PATH, 'images', split)
    all_files = sorted(os.listdir(img_dir))
    video_ids = sorted(set(f.split('_')[1] for f in all_files if '_' in f))
    rows, n_det, n_matched, t0 = [], 0, 0, time.time()
    for vi, v_id in enumerate(video_ids):
        xml_file = os.path.join(XML_PATH, f"video_{v_id}_attributes.xml")
        if not os.path.exists(xml_file):
            continue
        peds_meta = {p.get('id'): (int(p.get('decision_point')), int(p.get('crossing')))
                     for p in ET.parse(xml_file).getroot().findall('pedestrian')}
        gt = load_gt_boxes(v_id)
        for fname in [f for f in all_files if f.startswith(f"video_{v_id}_")]:
            frame_no = int(fname.split('_')[2][1:5])
            img = enhance_image(cv2.imread(os.path.join(img_dir, fname)))
            boxes = det.predict(img, conf=0.28, classes=[0], verbose=False)[0].boxes.xyxy.cpu().numpy()
            n_det += len(boxes)
            gts = gt.get(frame_no, [])
            if len(boxes) == 0 or len(gts) == 0:
                continue
            iou = iou_matrix(boxes, [g[1] for g in gts])
            keep = [(di, gts[gi][0]) for di, gi in zip(*linear_sum_assignment(-iou)) if iou[di, gi] >= IOU_THR]
            n_matched += len(keep)
            keep = [(di, pid) for di, pid in keep if pid in peds_meta]
            if not keep:
                continue
            full = cv2.resize(img, (1920, 1080))
            kx, ky, kc = crop_pose(pose, full, boxes[[di for di, _ in keep]])
            for i, (di, pid) in enumerate(keep):
                dp, crossing = peds_meta[pid]
                b = boxes[di]
                rows.append({'video_id': v_id, 'ped_id': pid, 'frame': frame_no,
                             'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3],
                             'kx': kx[i].tolist(), 'ky': ky[i].tolist(), 'kc': kc[i].tolist(),
                             'dp': dp, 'label': crossing})
        if (vi + 1) % 20 == 0:
            print(f"  {split}: {vi + 1}/{len(video_ids)} videos, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_pickle(os.path.join(OUT, f"pose_{split}.pkl"))
    found = (np.stack(df.kc.values).max(1) > 0).mean() if len(df) else 0
    print(f"Saved pose_{split}.pkl ({len(df)} rows, pose found in {found:.0%}) | detections={n_det}, "
          f"IoU-matched to GT={n_matched} ({n_matched / max(n_det, 1):.0%}) | {time.time() - t0:.0f}s", flush=True)


if __name__ == '__main__':
    det, pose = YOLO(MODEL_PATH), YOLO(POSE_MODEL)
    for split in ['train', 'val', 'test']:
        process_split(det, pose, split)
