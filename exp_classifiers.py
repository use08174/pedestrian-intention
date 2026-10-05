"""Compare classifiers for crossing-intention prediction on YOLO26-pose features.

Feature sets (per frame):
  bbox  : cx, cy, vel_x, vel_y, delta_area, aspect_ratio        (6, same as run_pipeline_fixed.py)
  pose  : 17 keypoints normalised to the box (x, y, conf) + keypoint velocity   (51 + 34 = 85)
  both  : bbox + pose                                             (91)
Settings:
  all       : every frame of the track
  before_dp : only frames up to the pedestrian's decision point (intention, not action)
Classifiers:
  window summary (last, mean, std, last-first) -> LogReg / PCA+SVM-RBF / RandomForest / HistGB
  raw 10-frame sequence -> LSTM / GRU / TCN / Transformer (3 seeds)

Metrics on test: sequence-level AUC / balanced acc / macro-F1, and pedestrian-level AUC / balanced acc
(window scores averaged per pedestrian; the test split has only ~15 'Stay' pedestrians).\nbalanced acc / F1 use the decision threshold that maximises balanced accuracy on the validation set.

Usage: python exp_classifiers.py [--cv]   (--cv: 5-fold video-grouped CV over train+val+test instead)
"""
import os
import sys
import time
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'outputs_pose')
SEQ_LEN, FRAME_STRIDE, KP_CONF = 10, 5, 0.3
SEEDS = [0, 1, 2]
CV = '--cv' in sys.argv
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BBOX = ['cx', 'cy', 'vel_x', 'vel_y', 'delta_area', 'aspect_ratio']
POSE = [f'k{i}_{c}' for i in range(17) for c in 'xyc'] + [f'k{i}_v{c}' for i in range(17) for c in 'xy']
FEATS = {'bbox': BBOX, 'pose': POSE, 'both': BBOX + POSE}


# ---------------- frame features ----------------
def frame_features(df):
    df = df[df.label.isin([0, 1])].sort_values(['video_id', 'ped_id', 'frame']).reset_index(drop=True)
    w, h = df.x2 - df.x1, df.y2 - df.y1
    df['cx'], df['cy'], df['aspect_ratio'] = (df.x1 + df.x2) / 2, (df.y1 + df.y2) / 2, w / h
    g = df.groupby(['video_id', 'ped_id'])
    gap = g.frame.diff()
    df['streak'] = (gap != FRAME_STRIDE).astype(int).groupby([df.video_id, df.ped_id]).cumsum()
    cont = (gap == FRAME_STRIDE).values
    df['vel_x'] = np.where(cont, df.cx - g.cx.shift(), 0)
    df['vel_y'] = np.where(cont, df.cy - g.cy.shift(), 0)
    df['delta_area'] = np.where(cont, (w * h) / ((w * h).groupby([df.video_id, df.ped_id]).shift() + 1e-6), 1)

    kx, ky, kc = (np.stack(df[c].values) for c in ('kx', 'ky', 'kc'))
    ok = kc >= KP_CONF
    nx = np.where(ok, (kx - df.x1.values[:, None]) / w.values[:, None], 0)
    ny = np.where(ok, (ky - df.y1.values[:, None]) / h.values[:, None], 0)
    prev = lambda a: np.vstack([np.zeros((1, 17)), a[:-1]])
    both_ok = ok & prev(ok).astype(bool) & cont[:, None]
    vx, vy = np.where(both_ok, nx - prev(nx), 0), np.where(both_ok, ny - prev(ny), 0)
    pose = np.concatenate([np.stack([nx, ny, kc], -1).reshape(len(df), -1), np.stack([vx, vy], -1).reshape(len(df), -1)], 1)
    df = pd.concat([df.drop(columns=['kx', 'ky', 'kc']), pd.DataFrame(pose, columns=POSE)], axis=1)
    return df


def make_windows(df, cols):
    X, y, grp = [], [], []
    for (v, p, _), g in df.groupby(['video_id', 'ped_id', 'streak'], sort=False):
        if len(g) < SEQ_LEN:
            continue
        f = g[cols].values.astype(np.float32)
        for i in range(len(g) - SEQ_LEN + 1):
            X.append(f[i:i + SEQ_LEN])
            y.append(g.label.iat[0])
            grp.append(f'{v}/{p}')
    return np.array(X).reshape(-1, SEQ_LEN, len(cols)), np.array(y), np.array(grp)


def summarize(X):
    return np.concatenate([X[:, -1], X.mean(1), X.std(1), X[:, -1] - X[:, 0]], 1)


# ---------------- classical models (window summary) ----------------
CLASSIC = {
    'LogReg': lambda s: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight='balanced', max_iter=3000)),
    'PCA+SVM': lambda s: make_pipeline(StandardScaler(), PCA(0.95, random_state=s), SVC(C=1.0, class_weight='balanced')),
    'RandomForest': lambda s: RandomForestClassifier(400, min_samples_leaf=5, class_weight='balanced_subsample', n_jobs=12, random_state=s),
    'HistGB': lambda s: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight='balanced', random_state=s),
}


def classic_scores(clf, X):
    return clf.decision_function(X) if hasattr(clf, 'decision_function') else clf.predict_proba(X)[:, 1] - 0.5


# ---------------- sequence models ----------------
class RNN(nn.Module):
    def __init__(self, d, cell):
        super().__init__()
        self.rnn = cell(d, 64, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc(self.rnn(x)[0][:, -1])


class TCN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Conv1d(64, 64, 3, padding=2, dilation=2), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Conv1d(64, 64, 3, padding=4, dilation=4), nn.ReLU())
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return self.fc(torch.cat([h.mean(2), h[:, :, -1]], 1))


class Transformer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.inp = nn.Linear(d, 64)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, 64))
        layer = nn.TransformerEncoderLayer(64, 4, 128, 0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        return self.fc(self.enc(self.inp(x) + self.pos).mean(1))


DEEP = {'LSTM': lambda d: RNN(d, nn.LSTM), 'GRU': lambda d: RNN(d, nn.GRU),
        'TCN': TCN, 'Transformer': Transformer}


def train_deep(name, Xtr, ytr, Xva, yva, seed, epochs=40):
    torch.manual_seed(seed); np.random.seed(seed)
    model = DEEP[name](Xtr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    w = np.where(ytr == 1, len(ytr) / (2 * ytr.sum()), len(ytr) / (2 * (1 - ytr).sum())).astype(np.float32)
    Xt, yt, wt = (torch.tensor(a).to(device) for a in (Xtr, ytr.astype(np.float32), w))
    Xv = torch.tensor(Xva).to(device)
    best, best_state, bad = -1, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = nn.functional.binary_cross_entropy_with_logits(model(Xt[b]).squeeze(1), yt[b], weight=wt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sv = predict_deep(model, Xv)
        auc = roc_auc_score(yva, sv) if len(np.unique(yva)) > 1 else ep  # no val positives: keep last epoch
        if auc > best:
            best, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(best_state)
    return model


def predict_deep(model, X):
    model.eval()
    X = X if torch.is_tensor(X) else torch.tensor(X).to(device)
    with torch.no_grad():
        return torch.cat([model(X[i:i + 4096]).squeeze(1) for i in range(0, len(X), 4096)]).cpu().numpy()


# ---------------- metrics ----------------
def val_threshold(y, s):
    """Threshold on the score that maximises balanced accuracy on the validation windows."""
    if len(np.unique(y)) < 2:
        return 0.0
    cand = np.quantile(s, np.linspace(0.01, 0.99, 99))
    return float(cand[np.argmax([balanced_accuracy_score(y, (s > t).astype(int)) for t in cand])])


def metrics(y, s, grp, thr):
    """s: score (logit / decision function); thr: decision threshold chosen on val."""
    p = (s > thr).astype(int)
    ped = pd.DataFrame({'g': grp, 'y': y, 's': s}).groupby('g').agg(y=('y', 'first'), s=('s', 'mean'))
    return {'seq_auc': roc_auc_score(y, s), 'seq_bacc': balanced_accuracy_score(y, p),
            'seq_f1': f1_score(y, p, average='macro'),
            'ped_auc': roc_auc_score(ped.y, ped.s), 'ped_bacc': balanced_accuracy_score(ped.y, (ped.s > thr).astype(int)),
            'n_seq': len(y), 'n_stay_ped': int((ped.y == 0).sum()), 'n_cross_ped': int((ped.y == 1).sum())}


def run_one(tr, va, te, fs, setting, rows, tag):
    cols = FEATS[fs]
    if setting == 'before_dp':
        tr, va, te = (d[d.frame <= d.dp] for d in (tr, va, te))
    sc = StandardScaler().fit(tr[cols].values)
    norm = lambda d: d.assign(**dict(zip(cols, sc.transform(d[cols].values).T)))
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, gte) = (make_windows(norm(d), cols) for d in (tr, va, te))
    for name, seed in product(list(CLASSIC) + list(DEEP), SEEDS):
        t0 = time.time()
        if name in CLASSIC:
            clf = CLASSIC[name](seed).fit(summarize(Xtr), ytr)
            sv, st = classic_scores(clf, summarize(Xva)), classic_scores(clf, summarize(Xte))
        else:
            model = train_deep(name, Xtr, ytr, Xva, yva, seed)
            sv, st = predict_deep(model, Xva), predict_deep(model, Xte)
        rows.append({'split': tag, 'features': fs, 'setting': setting, 'model': name, 'seed': seed,
                     **metrics(yte, st, gte, val_threshold(yva, sv)), 'sec': time.time() - t0})
        print({k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)


def main():
    data = {s: frame_features(pd.read_pickle(os.path.join(OUT, f'pose_{s}.pkl'))) for s in ['train', 'val', 'test']}
    rows, out = [], os.path.join(OUT, 'results_cv.csv' if CV else 'results.csv')
    for fs, setting in product(['bbox', 'pose', 'both'], ['all', 'before_dp']):
        if not CV:
            run_one(data['train'], data['val'], data['test'], fs, setting, rows, 'official')
        else:
            full = pd.concat(data.values(), ignore_index=True)
            vids = full.video_id.values
            for k, (tri, tei) in enumerate(GroupKFold(5).split(full, groups=vids)):
                trd, ted = full.iloc[tri], full.iloc[tei]
                # inner val split (by video) for deep-model early stopping
                tv = np.unique(trd.video_id.values)
                rng = np.random.RandomState(k)
                vv = set(rng.choice(tv, len(tv) // 5, replace=False))
                run_one(trd[~trd.video_id.isin(vv)], trd[trd.video_id.isin(vv)], ted, fs, setting, rows, f'fold{k}')
        pd.DataFrame(rows).to_csv(out, index=False)
    print('saved', out)


if __name__ == '__main__':
    main()
