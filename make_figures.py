"""Figures and tables for EXPERIMENTS.md -> docs/figs/*.png, docs/tables.md"""
import os
import xml.etree.ElementTree as ET

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, 'docs', 'figs')
os.makedirs(FIG, exist_ok=True)

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
FS_COLOR = {'bbox': '#2a78d6', 'pose': '#eb6834', 'both': '#1baf7a'}
FS_NAME = {'bbox': 'Bounding box', 'pose': 'Pose', 'both': 'Bounding box + pose'}
MODEL_NAME = {'LogReg': 'Logistic Regression', 'PCA+SVM': 'Support Vector Machine', 'RandomForest': 'Random Forest',
              'HistGB': 'Gradient Boosting', 'LSTM': 'Long Short-Term Memory', 'GRU': 'Gated Recurrent Unit',
              'TCN': 'Temporal Convolutional Network', 'Transformer': 'Transformer'}
SETTING_NAME = {'all': 'All frames', 'before_dp': 'Before decision point'}
MODELS = ['LogReg', 'PCA+SVM', 'RandomForest', 'HistGB', 'LSTM', 'GRU', 'TCN', 'Transformer']
plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.edgecolor': GRID, 'figure.facecolor': SURFACE,
                     'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})


def style(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='y', color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def detector_recall():
    d = [('YOLO26x-pose\n640 pixels, confidence 0.25', 0.41), ('YOLO26x-pose\n1280 pixels, confidence 0.1', 0.49),
         ('YOLO11x-pose\n1280 pixels, confidence 0.1', 0.50), ('YOLO26x-pose\n16:9 1920 pixels, confidence 0.1', 0.56),
         ('YOLO fine-tuned on JAAD\n(first-phase model)', 0.73)]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(d))[::-1]
    ax.barh(y, [v for _, v in d], height=0.55, color=['#9c9a92'] * 4 + [FS_COLOR['bbox']])
    for yi, (_, v) in zip(y, d):
        ax.text(v + 0.01, yi, f'{v:.2f}', va='center', color=INK)
    ax.set_yticks(y, [n for n, _ in d])
    ax.set_xlim(0, 0.85)
    ax.set_xlabel('Share of labelled pedestrians found (overlap ≥ 0.5, validation subset)')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'detector_recall.png'), dpi=160); plt.close(fig)


def baseline_fix():
    rows = [('Always "Cross"', 0.500, 0.483), ('Original notebook', 0.495, 0.49),
            ('Fixed', 0.843, 0.712), ('Fixed, before\ndecision point', 0.807, 0.605)]
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    x = np.arange(len(rows)); w = 0.36
    for k, (lab, c) in enumerate([('Balanced accuracy', FS_COLOR['bbox']), ('Macro-averaged F1 score', FS_COLOR['pose'])]):
        vals = [r[1 + k] for r in rows]
        ax.bar(x + (k - 0.5) * (w + 0.02), vals, w, color=c, label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 0.5) * (w + 0.02), v + 0.015, f'{v:.2f}', ha='center', fontsize=8.5, color=INK)
    ax.set_xticks(x, [r[0] for r in rows]); ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc='upper left', ncol=2); style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'baseline_fix.png'), dpi=160); plt.close(fig)


def model_bars(df, metric, fname, ylabel):
    agg = df.groupby(['setting', 'features', 'model'])[metric].agg(['mean', 'std']).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, setting, title in zip(axes, ['all', 'before_dp'], ['All frames', 'Before decision point']):
        x = np.arange(len(MODELS)); w = 0.26
        for k, fs in enumerate(['bbox', 'pose', 'both']):
            a = agg[(agg.setting == setting) & (agg.features == fs)].set_index('model').reindex(MODELS)
            ax.bar(x + (k - 1) * (w + 0.015), a['mean'].fillna(0), w, yerr=a['std'], color=FS_COLOR[fs],
                   error_kw={'ecolor': INK2, 'elinewidth': 0.8, 'capsize': 0}, label=FS_NAME[fs])
        ax.set_xticks(x, [MODEL_NAME[m] for m in MODELS], rotation=30, ha='right'); ax.set_title(title, color=INK, fontsize=11)
        ax.axhline(0.5, color=INK2, lw=0.8, ls='--')
        ax.set_ylim(0.3, 1.0); style(ax)
    axes[0].set_ylabel(ylabel)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(FIG, fname), dpi=160); plt.close(fig)


SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def pose_example():
    """Pose overlay for one crossing and one non-crossing pedestrian from the test split."""
    df = pd.read_pickle(os.path.join(ROOT, 'outputs_pose', 'pose_test.pkl'))
    df['h'] = df.y2 - df.y1
    df['nconf'] = df.kc.apply(lambda c: (np.array(c) > 0.5).sum())
    picks = []
    for lab in (1, 0):
        c = df[(df.label == lab) & (df.nconf >= 15)].sort_values('h', ascending=False)
        picks.append(c.iloc[len(c) // 4])
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, r in zip(axes, picks):
        img = cv2.imread(os.path.join(ROOT, 'data', 'vehic-ped-intuition', 'images', 'test',
                                      f"video_{r.video_id}_f{r.frame:04d}.jpg"))
        img = cv2.cvtColor(cv2.resize(img, (1920, 1080)), cv2.COLOR_BGR2RGB)
        sx, sy = 3.0, 1080 / 640
        kx, ky, kc = np.array(r.kx) * sx, np.array(r.ky) * sy, np.array(r.kc)
        x1, y1, x2, y2 = r.x1 * sx, r.y1 * sy, r.x2 * sx, r.y2 * sy
        pw, ph = (x2 - x1) * 0.8, (y2 - y1) * 0.35
        cx1, cy1 = int(max(x1 - pw, 0)), int(max(y1 - ph, 0))
        cx2, cy2 = int(min(x2 + pw, 1920)), int(min(y2 + ph, 1080))
        ax.imshow(img[cy1:cy2, cx1:cx2])
        ax.add_patch(plt.Rectangle((x1 - cx1, y1 - cy1), x2 - x1, y2 - y1, fill=False, ec='#ffffff', lw=1.5))
        for a, b in SKELETON:
            if kc[a] > 0.3 and kc[b] > 0.3:
                ax.plot([kx[a] - cx1, kx[b] - cx1], [ky[a] - cy1, ky[b] - cy1], color=FS_COLOR['pose'], lw=2)
        ok = kc > 0.3
        ax.scatter(kx[ok] - cx1, ky[ok] - cy1, s=14, color='#ffffff', edgecolors=FS_COLOR['pose'], linewidths=1.2, zorder=3)
        ax.set_title(f"{'Cross' if r.label == 1 else 'Stay'}  (video_{r.video_id}, frame {r.frame})", color=INK, fontsize=10)
        ax.axis('off')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'pose_example.png'), dpi=160); plt.close(fig)


def table(df, metric):
    """Compact markdown table: rows = models, columns = {All, Before DP} x {bbox, pose, both}; best per column in bold."""
    m = df.groupby(['model', 'setting', 'features'])[metric].mean()
    cols = [(st, fs) for st in ['all', 'before_dp'] for fs in ['bbox', 'pose', 'both']]
    best = {c: max(MODELS, key=lambda r: m.get((r, *c), -1)) for c in cols}
    out = ['| Model | All frames: bounding box | All frames: pose | All frames: bounding box + pose | Before decision point: bounding box | Before decision point: pose | Before decision point: bounding box + pose |',
           '|---|' + '---|' * 6]
    for r in MODELS:
        cells = ['–' if (r, *c) not in m.index else
                 f"**{m[(r, *c)]:.3f}**" if best[c] == r else f"{m[(r, *c)]:.3f}" for c in cols]
        out.append(f"| {MODEL_NAME[r]} | " + ' | '.join(cells) + ' |')
    return '\n'.join(out)


if __name__ == '__main__':
    detector_recall(); pose_example()
    md = []
    for f, name, tag in [('results.csv', 'Official test split', 'official'), ('results_cv.csv', '5-fold CV', 'cv')]:
        p = os.path.join(ROOT, 'outputs_pose', f)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        model_bars(df, 'seq_auc', f'models_{tag}_seq_auc.png', 'Area under the receiver operating\ncharacteristic curve')
        for metric in ['seq_auc', 'seq_bacc', 'ped_auc']:
            md.append(f'### {name} · {metric}\n\n' + table(df, metric) + '\n')
    open(os.path.join(ROOT, 'docs', 'tables.md'), 'w').write('\n'.join(md))
    print('done')
