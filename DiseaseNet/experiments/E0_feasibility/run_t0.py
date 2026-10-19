"""E0 / T0: frozen DINOv2 ViT-S/14 + linear classifier, per synthetic arm.

Strict synthetic-only setting: fixed hyper-parameters (LogisticRegression C=1, no tuning, no early
stopping), so no real label is used for any choice. For each arm with complete coverage
(every Stage-0 class has 3 parent assets rendered):
  synthetic IID  3-fold CV over parent assets (train on 2 assets/class, test on the held-out asset)
  real           train on all 3 assets/class, test on the locked real test cache
Reference arm PHOTO = the same 3 real parent photos/class with the same 2D variants (few-shot real).
usage: CUDA_VISIBLE_DEVICES=0 ~/Desktop/Research/.venv/bin/python experiments/E0_feasibility/run_t0.py
  -> results/E0/t0_summary.json, results/E0/t0_predictions.npz
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.training.features import extract  # noqa: E402
from src.evaluation import metrics as Mx  # noqa: E402

OUT = f"{ROOT}/results/E0"
ARMS = ["PHOTO", "U3D", "KB3D", "KB2D", "GEN2D"]


def clf():
    return LogisticRegression(C=1.0, max_iter=3000)


def main():
    os.makedirs(OUT, exist_ok=True)
    classes = sorted(c["class_id"] for c in json.load(open(f"{ROOT}/configs/datasets/stage0_classes.json"))["classes"])
    cix = {c: i for i, c in enumerate(classes)}
    syn = pd.concat([pd.read_csv(f"{ROOT}/data/synthetic/stage0/{m}") for m in ("manifest_3d.csv", "manifest_2d.csv")
                     if os.path.exists(f"{ROOT}/data/synthetic/stage0/{m}")])
    real = pd.read_csv(f"{ROOT}/data/real_splits/stage0_test.csv")
    Fr = extract(real.path.tolist()); yr = real.class_id.map(cix).values
    res, preds = {}, {}
    for arm in ARMS:
        S = syn[syn.arm == arm]
        cov = S.groupby("class_id").asset_id.nunique()
        if len(cov) < len(classes) or (cov < 3).any():
            res[arm] = dict(status=f"incomplete: {int((cov >= 3).sum())}/{len(classes)} classes with 3 assets"); continue
        Fs = extract(S.path.tolist()); ys = S.class_id.map(cix).values
        # synthetic IID: hold out the k-th asset of every class
        rank = S.groupby("class_id").asset_id.transform(lambda s: s.map({a: i for i, a in enumerate(sorted(s.unique()))})).values
        iid = []
        for k in range(3):
            tr, te = rank != k, rank == k
            m = clf().fit(Fs[tr], ys[tr]); p = m.predict(Fs[te])
            iid.append(Mx.summary(ys[te], p, m.predict_proba(Fs[te]), classes=list(range(len(classes)))))
        m = clf().fit(Fs, ys); pr = m.predict(Fr); P = m.predict_proba(Fr)
        rs = Mx.summary(yr, pr, P, classes=list(range(len(classes))))
        rs["macro_f1_ci"] = Mx.group_bootstrap(yr, pr, real.group.values, lambda a, b: Mx.summary(a, b)["macro_f1"])
        res[arm] = dict(n_train=int(len(S)), iid={k: float(np.mean([d[k] for d in iid])) for k in iid[0]}, iid_folds=iid,
                        real=rs, real_crop=Mx.crop_split(yr, pr, classes),
                        real_per_class_recall=dict(zip(classes, Mx.per_class_recall(yr, pr, len(classes)))))
        preds[arm] = pr
        print(f"{arm:6s} IID macro-F1 {res[arm]['iid']['macro_f1']:.3f} | REAL macro-F1 {rs['macro_f1']:.3f} "
              f"[{rs['macro_f1_ci'][0]:.3f}, {rs['macro_f1_ci'][1]:.3f}] top1 {rs['top1']:.3f} crop-acc {res[arm]['real_crop']['crop_acc']:.3f}", flush=True)
    comp = {}
    for a in preds:
        for b in preds:
            if a < b:
                d, ci = Mx.paired_group_bootstrap(yr, preds[a], preds[b], real.group.values, lambda y, p: Mx.summary(y, p)["macro_f1"])
                comp[f"{a}-{b}"] = dict(diff=d, ci=ci)
    json.dump(dict(classes=classes, n_real_test=int(len(real)), arms=res, paired_real_macro_f1=comp,
                   setting="strict synthetic-only: fixed hyper-parameters, no real labels used for selection; pretrained DINOv2 (generic real images)"),
              open(f"{OUT}/t0_summary.json", "w"), indent=1)
    np.savez(f"{OUT}/t0_predictions.npz", y=yr, **{k: v for k, v in preds.items()})


if __name__ == "__main__":
    main()
