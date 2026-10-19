"""Stage 1 experiment harness: training sets from the Stage-1 image store, frozen-feature probes (T0/T1-linear),
evaluation on the DEV benchmark (= the Stage-0 test, now exploratory) and, behind a guard, on the FINAL test.

Evaluation sets (all real SAGE, locked partitions; group-level bootstrap):
  dev      data/real_splits/stage0_test.csv  (4,202 photos; seen during Stage 0 -> exploratory/diagnostic only)
  final    data/real_splits/stage1_final_<cohort>.csv from the untouched `val` partition. Reading it requires
           final=True and appends a line to results/stage1/FINAL_ACCESS_LOG.csv (who/what/when). Only frozen
           configurations from DISEASENET_STAGE1_MASTER_PROTOCOL.md may be evaluated on it.
  lopo     synthetic leave-one-parent-out: train on the other parents of every class, test on the held-out parent's
           images (strict-S0 model selection uses only this).
Probe = Stage-0 T0: L2-normalised frozen DINOv2 ViT-S/14 CLS + LogisticRegression(C=1, lbfgs), fixed, no tuning.
"""
import csv
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.evaluation import metrics as Mx  # noqa: E402
from src.training.features import extract  # noqa: E402

S1 = f"{ROOT}/data/synthetic/stage1"
RES = f"{ROOT}/results/stage1"


def classes20():
    return sorted(c["class_id"] for c in json.load(open(f"{ROOT}/configs/datasets/stage0_classes.json"))["classes"])


def manifest(arm):
    """Stage-1 arm manifest; 'S0:<ARM>' gives the original Stage-0 rows (audited record)."""
    if arm.startswith("S0:"):
        a = arm[3:]
        d = pd.concat([pd.read_csv(f"{ROOT}/data/synthetic/stage0/{m}") for m in ("manifest_3d.csv", "manifest_2d.csv")])
        d = d[d.arm == a].copy(); d["arm"] = arm; d["edit_seed"] = 0; d["view"] = d.view.map(lambda v: f"v{v}")
        return d
    return pd.read_csv(f"{S1}/manifest_{arm}.csv")


def dev_set():
    return pd.read_csv(f"{ROOT}/data/real_splits/stage0_test.csv")


def final_set(cohort, purpose):
    p = f"{ROOT}/data/real_splits/stage1_final_{cohort}.csv"
    os.makedirs(RES, exist_ok=True); new = not os.path.exists(f"{RES}/FINAL_ACCESS_LOG.csv")
    with open(f"{RES}/FINAL_ACCESS_LOG.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "cohort", "purpose"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), cohort, purpose])
    return pd.read_csv(p)


def probe_fit(Xtr, ytr):
    return LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, ytr)


def evaluate(y, P, groups, n_classes, classes):
    pr = P.argmax(1); rs = Mx.summary(y, pr, P, classes=list(range(n_classes)))
    rs["macro_f1_ci"] = Mx.group_bootstrap(y, pr, groups, lambda a, b: Mx.summary(a, b)["macro_f1"])
    rs["per_class_recall"] = dict(zip(classes, Mx.per_class_recall(y, pr, n_classes)))
    rs["crop"] = Mx.crop_split(y, pr, classes)
    return rs


def run_probe(name, train, classes, out_dir, evals=("dev",), lopo=True, final_cohort=None, final_purpose=""):
    """train: DataFrame with path, class_id, root_sha1. Saves <out_dir>/<name>.json + .npz (per-image probabilities)."""
    os.makedirs(out_dir, exist_ok=True); cix = {c: i for i, c in enumerate(classes)}
    train = train[train.class_id.isin(classes)]
    X = extract(train.path.tolist()); y = train.class_id.map(cix).values
    res = dict(name=name, n_train=int(len(train)), n_parents=int(train.root_sha1.nunique()),
               parents_per_class=train.groupby("class_id").root_sha1.nunique().describe()[["min", "mean", "max"]].round(2).to_dict())
    store = {}
    if lopo:                                                # synthetic leave-one-parent-out (rank within class)
        rk = train.groupby("class_id").root_sha1.transform(lambda s: s.map({r: i for i, r in enumerate(sorted(s.unique()))})).values
        K = int(train.groupby("class_id").root_sha1.nunique().min()); f1s = []
        for k in range(min(K, 3)):
            tr, te = rk != k, rk == k
            m = probe_fit(X[tr], y[tr]); f1s.append(Mx.summary(y[te], m.predict(X[te]), classes=list(range(len(classes))))["macro_f1"])
        res["lopo_macro_f1"] = float(np.mean(f1s)); res["lopo_folds"] = f1s
    m = probe_fit(X, y)
    sets = {e: dev_set() for e in evals if e == "dev"}
    if final_cohort:
        sets["final"] = final_set(final_cohort, final_purpose or name)
    for e, D in sets.items():
        D = D[D.class_id.isin(classes)]
        P = np.zeros((len(D), len(classes)), np.float32); P[:, m.classes_] = m.predict_proba(extract(D.path.tolist()))
        yy = D.class_id.map(cix).values
        res[e] = evaluate(yy, P, D.group.values, len(classes), classes); store[f"{e}_proba"] = P; store[f"{e}_y"] = yy; store[f"{e}_g"] = D.group.values
    json.dump(res, open(f"{out_dir}/{name}.json", "w"), indent=1)
    np.savez_compressed(f"{out_dir}/{name}.npz", **store)
    return res


def paired(out_dir, a, b, split="dev", n=2000, seed=0, out_dir_b=None):
    """Paired group-bootstrap of macro-F1(a) - macro-F1(b) from saved predictions (same test images).
    Returns dict(diff, ci, p) with a two-sided bootstrap p = 2 * min(P(d <= 0), P(d >= 0))."""
    A, B = np.load(f"{out_dir}/{a}.npz"), np.load(f"{out_dir_b or out_dir}/{b}.npz")
    y, g = A[f"{split}_y"], A[f"{split}_g"]; assert (y == B[f"{split}_y"]).all()
    pa, pb = A[f"{split}_proba"].argmax(1), B[f"{split}_proba"].argmax(1); k = A[f"{split}_proba"].shape[1]
    f = lambda u, v: Mx.summary(u, v, classes=list(range(k)))["macro_f1"]  # noqa: E731
    ug = np.unique(g); idx = {x: np.flatnonzero(g == x) for x in ug}; rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        s_ = np.concatenate([idx[x] for x in rng.choice(ug, len(ug), replace=True)]); d.append(f(y[s_], pa[s_]) - f(y[s_], pb[s_]))
    d = np.array(d)
    return dict(diff=float(f(y, pa) - f(y, pb)), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                p=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))


def holm(pvals):
    """Holm step-down adjusted p-values (dict name -> p)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1]); m = len(items); adj, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p)); adj[k] = run
    return adj
