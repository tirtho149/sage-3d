"""Stage 1, step 8 (Adapted S1), frozen-feature version: synthetic training set + k real shots/class vs k real shots only.

Real shots come from the `adapt` partition, split 50/50 by near-duplicate group into adapt-shots and adapt-val
(seed 20261016). For each k in {0,1,4,8,16} and 5 shot draws: logistic regression (C=1) on
  SYN+REAL  the synthetic set + k shots/class, real shots up-weighted by w in {1, 10, 100} chosen on adapt-val
  REAL      the k shots only (k >= 1)  — the generic-pretrained baseline with identical shots and budget
  REAL+3    the k shots + the class's 3 Stage-0 parent photos (no augmentation): same REAL-PHOTO budget as the
            synthetic arms, which are all derived from those 3 parents (budget-matched baseline)
Note: v0 (results/stage1/fewshot_v0) used w <= 100 and selected the grid edge every time; v1 adds w = 1000.
Evaluated on DEV. k = 0 for SYN+REAL is the strict-S0 probe. Classes with fewer adapt-shot photos than k use all.
usage: CUDA_VISIBLE_DEVICES=0 ~/Desktop/Research/.venv/bin/python experiments/S1/run_fewshot.py SET [SET ...]
  -> results/stage1/fewshot/<SET>.json (+ REAL.json)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from src.evaluation import metrics as Mx  # noqa: E402
from src.s1 import harness as Hn  # noqa: E402
from src.s1.sets import make  # noqa: E402
from src.training.features import extract  # noqa: E402

KS, DRAWS, WS = [0, 1, 4, 8, 16], 5, [1.0, 10.0, 100.0, 1000.0]
OUT = f"{ROOT}/results/stage1/fewshot"


def adapt_split(classes):
    A = pd.read_csv(f"{ROOT}/data/real_splits/stage1_adapt_all.csv"); A = A[A.class_id.isin(classes)]
    g = np.sort(A.group.unique()); rng = np.random.default_rng(20261016); val_g = set(rng.choice(g, len(g) // 2, replace=False))
    return A[~A.group.isin(val_g)], A[A.group.isin(val_g)]


def f1(y, p, n):
    return Mx.summary(y, p, classes=list(range(n)))["macro_f1"]


def main():
    os.makedirs(OUT, exist_ok=True)
    C = Hn.classes20(); n = len(C); cix = {c: i for i, c in enumerate(C)}
    shots, aval = adapt_split(C); D = Hn.dev_set(); D = D[D.class_id.isin(C)]
    Xs_all, ys_all = extract(shots.path.tolist()), shots.class_id.map(cix).values
    Xv, yv = extract(aval.path.tolist()), aval.class_id.map(cix).values
    Xd, yd = extract(D.path.tolist()), D.class_id.map(cix).values
    draws = {}
    for k in KS[1:]:
        for d in range(DRAWS):
            rng = np.random.default_rng(1000 * k + d)
            draws[(k, d)] = np.concatenate([rng.permutation(np.flatnonzero(ys_all == c))[:k] for c in range(n)])
    if not os.path.exists(f"{OUT}/REAL.json"):
        res = {}
        for (k, d), ix in draws.items():
            m = LogisticRegression(C=1.0, max_iter=3000).fit(Xs_all[ix], ys_all[ix])
            P = np.zeros((len(Xd), n)); P[:, m.classes_] = m.predict_proba(Xd)
            res.setdefault(str(k), []).append(f1(yd, P.argmax(1), n))
        json.dump(dict(set="REAL", dev_macro_f1=res, n_adapt_shots_pool=int(len(shots)), n_adapt_val=int(len(aval))), open(f"{OUT}/REAL.json", "w"), indent=1)
        print("REAL", {k: round(np.mean(v), 3) for k, v in res.items()}, flush=True)
    if not os.path.exists(f"{OUT}/REAL+3.json"):
        sys.path.insert(0, ROOT)
        from src.generation.registry import assets
        A = assets("stage0"); A = A[A.class_id.isin(C)]
        Xp, yp = extract([os.path.relpath(p, ROOT) if os.path.isabs(p) else p for p in A.image]), A.class_id.map(cix).values
        res = {}
        for k in KS:
            for d in (range(DRAWS) if k else [0]):
                ix = draws.get((k, d), np.array([], int))
                m = LogisticRegression(C=1.0, max_iter=3000).fit(np.concatenate([Xp, Xs_all[ix]]), np.concatenate([yp, ys_all[ix]]))
                P = np.zeros((len(Xd), n)); P[:, m.classes_] = m.predict_proba(Xd)
                res.setdefault(str(k), []).append(f1(yd, P.argmax(1), n))
        json.dump(dict(set="REAL+3", dev_macro_f1=res), open(f"{OUT}/REAL+3.json", "w"), indent=1)
        print("REAL+3", {k: round(np.mean(v), 3) for k, v in res.items()}, flush=True)
    for name in sys.argv[1:]:
        if os.path.exists(f"{OUT}/{name}.json"):
            continue
        S = make(name); S = S[S.class_id.isin(C)]
        X, y = extract(S.path.tolist()), S.class_id.map(cix).values
        res, wsel = {}, {}
        for k in KS:
            for d in (range(DRAWS) if k else [0]):
                ix = draws.get((k, d), np.array([], int))
                best = None
                for w in (WS if k else [1.0]):
                    Xt = np.concatenate([X, Xs_all[ix]]); yt = np.concatenate([y, ys_all[ix]])
                    sw = np.concatenate([np.ones(len(y)), np.full(len(ix), w)])
                    m = LogisticRegression(C=1.0, max_iter=3000).fit(Xt, yt, sample_weight=sw)
                    Pv = np.zeros((len(Xv), n)); Pv[:, m.classes_] = m.predict_proba(Xv); fv = f1(yv, Pv.argmax(1), n)
                    if best is None or fv > best[0]:
                        best = (fv, w, m)
                m = best[2]; P = np.zeros((len(Xd), n)); P[:, m.classes_] = m.predict_proba(Xd)
                res.setdefault(str(k), []).append(f1(yd, P.argmax(1), n)); wsel.setdefault(str(k), []).append(best[1])
        json.dump(dict(set=name, dev_macro_f1=res, selected_w=wsel), open(f"{OUT}/{name}.json", "w"), indent=1)
        print(name, {k: round(np.mean(v), 3) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
