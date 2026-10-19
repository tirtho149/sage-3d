"""Statistics from protocol §7. Unit = independent specimen/test image; clusters respected."""
from __future__ import annotations
import numpy as np
from scipy import stats as st


def cluster_bootstrap_diff(a, b, clusters=None, n=10000, seed=0, fn=np.mean):
    """Paired difference fn(a)-fn(b) with a cluster bootstrap 95% percentile CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    clusters = np.arange(len(a)) if clusters is None else np.asarray(clusters)
    uc = np.unique(clusters); idx = {c: np.where(clusters == c)[0] for c in uc}
    rng = np.random.default_rng(seed); boots = np.empty(n)
    for i in range(n):
        s = np.concatenate([idx[c] for c in rng.choice(uc, len(uc), replace=True)])
        boots[i] = fn(a[s]) - fn(b[s])
    return {"diff": float(fn(a) - fn(b)), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_units": int(len(a)), "n_clusters": int(len(uc))}


def mcnemar_exact(correct_a, correct_b):
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    p = st.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return {"a_only": n10, "b_only": n01, "p": float(p)}


def wilcoxon_paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b; d = d[~np.isnan(d)]
    if len(d) < 2 or np.allclose(d, 0):
        return {"n": int(len(d)), "p": float("nan"), "rank_biserial": float("nan"), "median_diff": float(np.median(d)) if len(d) else float("nan")}
    r = st.wilcoxon(d)
    ranks = st.rankdata(np.abs(d)); rb = (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum()
    return {"n": int(len(d)), "p": float(r.pvalue), "rank_biserial": float(rb), "median_diff": float(np.median(d))}


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=lambda k: pvals[k]); m = len(keys); out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k])); out[k] = run
    return out


def macro_f1(y_true, y_pred):
    labs = np.unique(np.concatenate([y_true, y_pred])); f = []
    for l in labs:
        tp = np.sum((y_pred == l) & (y_true == l)); fp = np.sum((y_pred == l) & (y_true != l)); fn = np.sum((y_pred != l) & (y_true == l))
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f))


def design_effect_n(n_discordant_pairs_needed: int, m: float, icc: float) -> int:
    return int(np.ceil(n_discordant_pairs_needed * (1 + (m - 1) * icc)))
