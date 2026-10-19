"""Metrics for DiseaseNet (primary: macro-F1 on the locked real test set).

All CIs resample near-duplicate GROUPS of the real test set (the split unit), not images.
"""
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score


def summary(y, pred, proba=None, classes=None):
    out = dict(macro_f1=float(f1_score(y, pred, average="macro", labels=classes)),
               top1=float(np.mean(np.asarray(y) == np.asarray(pred))),
               bal_acc=float(balanced_accuracy_score(y, pred)))
    if proba is not None:
        top5 = np.argsort(-proba, 1)[:, :5]
        out["top5"] = float(np.mean([yy in t for yy, t in zip(y, top5)]))
    return out


def crop_split(y, pred, class_names):
    """Share of errors that get the crop right (disease confusion) vs wrong crop."""
    crop = np.array([c.split("::")[0] for c in class_names])
    err = np.asarray(y) != np.asarray(pred)
    same_crop = crop[np.asarray(y)] == crop[np.asarray(pred)]
    return dict(crop_acc=float(np.mean(same_crop)), errors=int(err.sum()),
                errors_right_crop=int((err & same_crop).sum()), errors_wrong_crop=int((err & ~same_crop).sum()))


def per_class_recall(y, pred, n_classes):
    y, pred = np.asarray(y), np.asarray(pred)
    return [float(np.mean(pred[y == k] == k)) if np.any(y == k) else float("nan") for k in range(n_classes)]


def group_bootstrap(y, pred, groups, fn, n=2000, seed=0):
    y, pred, groups = map(np.asarray, (y, pred, groups))
    ug = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        vals.append(fn(y[s], pred[s]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired_group_bootstrap(y, pa, pb, groups, fn, n=2000, seed=0):
    """CI of fn(a) - fn(b) on the same test images, resampling groups."""
    y, pa, pb, groups = map(np.asarray, (y, pa, pb, groups))
    ug = np.unique(groups); idx = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug), replace=True)])
        d.append(fn(y[s], pa[s]) - fn(y[s], pb[s]))
    return float(fn(y, pa) - fn(y, pb)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
