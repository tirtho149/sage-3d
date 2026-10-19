"""Render-to-real gap: error analysis of the minimal downstream test (review item A12).

Uses only cached artifacts of the pre-registered run (results/downstream_minimal/):
  features.npz (frozen DINOv2 features), image manifests, per_draw_metrics.csv.
Three questions, all descriptive (no new hypothesis test, nothing tuned):
  1. Distance: cosine distance from each image of a condition to its nearest REAL image of
     the same pilot class (build pool). Reference = real eval images of that class (the
     real-to-real spread). A synthetic set far beyond the reference is off the real manifold.
  2. Separability: 5-fold logistic-regression AUC for "real pilot-class photo vs this
     condition". AUC near 0.5 = indistinguishable; near 1.0 = trivially separable (a gap).
  3. Error mode: paired per-draw change vs real-only training (cond - R) in pilot-class
     recall and in the false-positive rate on the other classes, per k.
Background is a controlled factor: render conditions exist with natural and white
backgrounds (bgmode), so its share of the gap can be read off directly.

usage: ~/Desktop/Research/.venv/bin/python -m src.downstream.gap_analysis
  -> results/downstream_minimal/gap_analysis.json, gap_analysis.{pdf,png}, figures/slides/gap_numbers.tex
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .common import OUT

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONDS = ["S", "KB2D", "D", "GEN3D", "KB3D", "J", "F"]
LABEL = {"S": "source photo", "KB2D": "2D KB edit", "D": "render, unedited", "GEN3D": "render, generic 3D",
         "KB3D": "render, KB 3D", "J": "render, geometry", "F": "render, GE+KB"}


def load():
    z = np.load(OUT / "features.npz")
    fmap = {p: i for i, p in enumerate(z["paths"])}
    F = z["feats"].astype(np.float32)
    F /= np.linalg.norm(F, axis=1, keepdims=True) + 1e-8
    real = pd.read_csv(OUT / "image_manifest_real.csv")
    syn = pd.read_csv(OUT / "image_manifest_syn.csv")
    real["fi"] = real.path.map(fmap)
    syn["fi"] = syn.path.map(fmap)
    return F, real.dropna(subset=["fi"]), syn.dropna(subset=["fi"])


def nn_dist(F, q, ref):
    if len(q) == 0 or len(ref) == 0:
        return np.array([])
    S = F[q] @ F[ref].T
    return 1 - S.max(1)


def auc(F, a, b, seed=0):
    n = min(len(a), len(b))
    if n < 10:
        return float("nan")
    rng = np.random.default_rng(seed)
    a = rng.choice(a, n, replace=False) if len(a) > n else a
    b = rng.choice(b, n, replace=False) if len(b) > n else b
    X = np.concatenate([F[a], F[b]]); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(LogisticRegression(max_iter=2000, C=1.0), X, y, cv=cv, scoring="roc_auc").mean())


def main():
    F, real, syn = load()
    out = {"distance": {}, "separability": {}, "error_mode": {}, "background": {}}
    pil = real[real.is_pilot_class.astype(bool)]
    build = pil[pil.split == "build"]
    evalr = pil[pil.split.isin(["test", "val"])]
    rows = []
    for crop in sorted(pil.crop.unique()):
        ref = build[build.crop == crop].fi.astype(int).values
        q_real = evalr[evalr.crop == crop].fi.astype(int).values
        rows += [dict(crop=crop, cond="REAL", bg="natural", d=v) for v in nn_dist(F, q_real, ref)]
        for c in CONDS:
            for bg in syn.bgmode.unique():
                q = syn[(syn.crop == crop) & (syn.cond == c) & (syn.bgmode == bg)].fi.astype(int).values
                rows += [dict(crop=crop, cond=c, bg=bg, d=v) for v in nn_dist(F, q, ref)]
    D = pd.DataFrame(rows)
    for (c, bg), g in D.groupby(["cond", "bg"]):
        out["distance"][f"{c}|{bg}"] = dict(median=float(g.d.median()), q25=float(g.d.quantile(.25)),
                                             q75=float(g.d.quantile(.75)), n=int(len(g)))
    realfi = pil.fi.astype(int).values
    for c in CONDS:
        for bg in syn.bgmode.unique():
            s = syn[(syn.cond == c) & (syn.bgmode == bg)].fi.astype(int).values
            if len(s):
                out["separability"][f"{c}|{bg}"] = auc(F, realfi, s)
    m = pd.read_csv(OUT / "per_draw_metrics.csv")
    base = m[m.cond == "R"].set_index(["crop", "k", "draw"])
    for c in ["S", "KB2D", "D", "KB3D", "D_white", "KB3D_white"]:
        x = m[m.cond == c].set_index(["crop", "k", "draw"])
        if x.empty:
            continue
        d = (x[["pilot_recall", "fpr_other", "macro_f1"]] - base[["pilot_recall", "fpr_other", "macro_f1"]]).dropna()
        out["error_mode"][c] = {int(k): {col: float(g[col].mean()) for col in d.columns}
                                for k, g in d.groupby(level="k")}
    for c in ["D", "KB3D"]:
        nat, wh = out["distance"].get(f"{c}|natural"), out["distance"].get(f"{c}|white")
        if nat and wh:
            out["background"][c] = dict(natural=nat["median"], white=wh["median"])
    json.dump(out, open(OUT / "gap_analysis.json", "w"), indent=1)
    figure(D, out)
    macros(out)
    print(json.dumps({k: out[k] for k in ("separability", "background")}, indent=1))


def figure(D, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["REAL"] + CONDS
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0), gridspec_kw=dict(width_ratios=[1.6, 1, 1.2]))
    nat = D[D.bg == "natural"]
    data = [nat[nat.cond == c].d.values for c in order]
    ax[0].boxplot(data, vert=False, showfliers=False, widths=0.6)
    ax[0].set_yticks(range(1, len(order) + 1), ["real eval (ref.)"] + [LABEL[c] for c in CONDS], fontsize=7)
    ax[0].axvline(np.median(data[0]), color="g", ls="--", lw=0.8)
    ax[0].set_xlabel("cosine distance to nearest real\npilot-class photo (DINOv2)", fontsize=7)
    ax[0].set_title("(a) distance to the real manifold", fontsize=8)
    sep = out["separability"]
    labs = [c for c in CONDS if f"{c}|natural" in sep]
    ax[1].barh(range(len(labs)), [sep[f"{c}|natural"] for c in labs], color="0.55", label="natural bg")
    w = [sep.get(f"{c}|white", np.nan) for c in labs]
    ax[1].scatter(w, range(len(labs)), color="crimson", s=12, zorder=3, label="white bg")
    ax[1].set_yticks(range(len(labs)), [LABEL[c] for c in labs], fontsize=7)
    ax[1].axvline(0.5, color="k", lw=0.6); ax[1].set_xlim(0.4, 1.0)
    ax[1].set_xlabel("real-vs-synthetic AUC", fontsize=7); ax[1].legend(fontsize=6, loc="lower right")
    ax[1].set_title("(b) separability", fontsize=8)
    em = out["error_mode"]
    ks = sorted({k for v in em.values() for k in v})
    for c, col in [("KB3D", "crimson"), ("D", "0.4"), ("KB2D", "tab:blue")]:
        if c in em:
            ax[2].plot(ks, [em[c][k]["pilot_recall"] for k in ks], "-o", color=col, ms=3, label=f"{LABEL[c]}: recall")
            ax[2].plot(ks, [em[c][k]["fpr_other"] for k in ks], "--s", color=col, ms=3, label=f"{LABEL[c]}: FPR other")
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xscale("log", base=2); ax[2].set_xticks(ks, [str(k) for k in ks])
    ax[2].set_xlabel("k real images per class", fontsize=7); ax[2].set_ylabel("change vs real-only", fontsize=7)
    ax[2].legend(fontsize=5, loc="best"); ax[2].set_title("(c) error mode", fontsize=8)
    for a in ax:
        a.tick_params(labelsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"gap_analysis.{ext}", dpi=200)
        fig.savefig(f"{R}/figures/slides/gap_analysis.{ext}", dpi=200)


def macros(out):
    d, s, e = out["distance"], out["separability"], out["error_mode"]
    M = {"GapReal": f"{d['REAL|natural']['median']:.2f}",
         "GapS": f"{d['S|natural']['median']:.2f}",
         "GapKBtwoD": f"{d['KB2D|natural']['median']:.2f}",
         "GapD": f"{d['D|natural']['median']:.2f}",
         "GapKBthreeD": f"{d['KB3D|natural']['median']:.2f}",
         "AucS": f"{s.get('S|natural', float('nan')):.2f}",
         "AucKBtwoD": f"{s.get('KB2D|natural', float('nan')):.2f}",
         "AucD": f"{s.get('D|natural', float('nan')):.2f}",
         "AucKBthreeD": f"{s.get('KB3D|natural', float('nan')):.2f}"}
    if "KB3D|white" in s:
        M["AucKBthreeDwhite"] = f"{s['KB3D|white']:.2f}"
    for c, tag in [("KB3D", "KBthreeD"), ("D", "D")]:
        if c in e and 4 in e[c]:
            M[f"Rec{tag}Four"] = f"{100 * e[c][4]['pilot_recall']:+.1f}"
            M[f"Fpr{tag}Four"] = f"{100 * e[c][4]['fpr_other']:+.1f}"
    tex = ["% generated by src/downstream/gap_analysis.py from results/downstream_minimal -- do not edit by hand"]
    tex += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    open(f"{R}/figures/slides/gap_numbers.tex", "w").write("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
