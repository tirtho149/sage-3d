"""Advisor-deck figure + macros for DiseaseNet Stage 0, from result files only (CPU).

Reads results/E0/t0_summary.json, t0_predictions.npz, t3_<arm>_{real,fold0}_s0.{json,npz}, the locked real test
csv (bootstrap groups) and the Stage-0 manifests. Adds paired T3 contrasts with the same group bootstrap as T0
(src/evaluation/metrics.paired_group_bootstrap, 2,000 resamples) -> results/E0/t3_paired.json.
Writes figures/deck/stage0_real.pdf and figures/deck/diseasenet_numbers.tex.
Deck (Overleaf 6ac80070): copy the PDF to figs/results/ and the .tex to generated/, then push.
usage: ~/Desktop/Research/.venv/bin/python scripts/deck_stage0.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.evaluation import metrics as Mx  # noqa: E402

E0, OUT = f"{ROOT}/results/E0", f"{ROOT}/figures/deck"
os.makedirs(OUT, exist_ok=True)
ARMS = ["PHOTO", "KB2D", "GEN2D", "KB3D", "U3D"]
LAB = {"PHOTO": "Real photos\n(3/class)", "KB2D": "2D edit,\nKB prompt", "GEN2D": "2D edit,\ngeneric",
       "KB3D": "3D render,\nKB edit", "U3D": "3D render,\nunedited"}
CARD, GRAY = "#C8102E", "#707372"

t0 = json.load(open(f"{E0}/t0_summary.json"))
# T3 seeds present for every arm and both evaluations (seed 0 first)
SEEDS = [k for k in range(10) if all(os.path.exists(f"{E0}/t3_{a}_{e}_s{k}.json") for a in ARMS for e in ("real", "fold0"))]
t3s = {(a, e, k): json.load(open(f"{E0}/t3_{a}_{e}_s{k}.json")) for a in ARMS for e in ("real", "fold0") for k in SEEDS}
t3 = {(a, e): t3s[(a, e, 0)] for a in ARMS for e in ("real", "fold0")}
t3m = {a: float(np.mean([t3s[(a, "real", k)]["metrics"]["macro_f1"] for k in SEEDS])) for a in ARMS}
real = pd.read_csv(f"{ROOT}/data/real_splits/stage0_test.csv")

# ---- paired T3 contrasts on the real test set (argmax of saved probabilities), per seed
real_g = real.group.values
f1 = lambda u, v: Mx.summary(u, v)["macro_f1"]  # noqa: E731
PAIRS = [("KB3D", "U3D"), ("KB3D", "PHOTO"), ("KB2D", "PHOTO"), ("GEN2D", "KB2D"), ("KB2D", "KB3D")]
t3p = {f"{a}-{b}": {} for a, b in PAIRS}
for k in SEEDS:
    P = {a: np.load(f"{E0}/t3_{a}_real_s{k}.npz") for a in ARMS}
    y = P["PHOTO"]["y"]
    assert all((P[a]["y"] == y).all() for a in ARMS) and len(y) == len(real)
    pred = {a: P[a]["proba"].argmax(1) for a in ARMS}
    for a, b in PAIRS:
        d, ci = Mx.paired_group_bootstrap(y, pred[a], pred[b], real_g, f1)
        t3p[f"{a}-{b}"][f"s{k}"] = dict(diff=d, ci=ci)
for v in t3p.values():
    ds = [v[f"s{k}"]["diff"] for k in SEEDS]
    v["mean_diff"] = float(np.mean(ds)); v["min_diff"] = float(min(ds)); v["max_diff"] = float(max(ds))
    v["seeds_ci_excl0_same_sign"] = int(sum((v[f"s{k}"]["ci"][0] > 0) if v["mean_diff"] > 0 else (v[f"s{k}"]["ci"][1] < 0) for k in SEEDS))
json.dump(dict(seeds=SEEDS, pairs=t3p), open(f"{E0}/t3_paired.json", "w"), indent=1)

# ---- figure: real-test macro-F1 with 95% CI, two protocols
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
fig, ax = plt.subplots(figsize=(4.6, 2.5))
x = np.arange(len(ARMS))
for k, (name, col, mk, off) in enumerate([("T0 frozen DINOv2 + linear probe", CARD, "o", -0.12),
                                          ("T3 full fine-tune", GRAY, "s", 0.12)]):
    if k == 0:
        v = [t0["arms"][a]["real"]["macro_f1"] for a in ARMS]
        ci = [t0["arms"][a]["real"]["macro_f1_ci"] for a in ARMS]
        err = np.array([[m - c[0], c[1] - m] for m, c in zip(v, ci)]).T
        ax.errorbar(x + off, v, yerr=err, fmt=mk, color=col, ms=4.5, lw=1.0, capsize=2, label=name + " (95% CI)", zorder=3)
    else:
        v = [t3m[a] for a in ARMS]
        for i, a in enumerate(ARMS):
            ax.scatter(np.full(len(SEEDS), x[i] + off), [t3s[(a, "real", kk)]["metrics"]["macro_f1"] for kk in SEEDS],
                       marker="_", s=40, color=col, lw=1.0, zorder=3)
        ax.scatter(x + off, v, marker=mk, s=22, color=col, zorder=4,
                   label=name + (f" (mean of {len(SEEDS)} seeds; ticks = seeds)" if len(SEEDS) > 1 else " (95% CI)"))
        if len(SEEDS) == 1:
            ci = [t3[(a, "real")]["metrics"]["macro_f1_ci"] for a in ARMS]
            ax.errorbar(x + off, v, yerr=np.array([[m - c[0], c[1] - m] for m, c in zip(v, ci)]).T, fmt="none", color=col, lw=1.0, capsize=2)
    for xi, m in zip(x + off, v):
        ax.text(xi + (-0.07 if k == 0 else 0.07), m, f"{m:.2f}", fontsize=5.6, color="#333333",
                ha="right" if k == 0 else "left", va="center")
ax.axvspan(2.5, 4.5, color="#eeeeee", zorder=0)
ax.text(3.5, 0.715, "3D renders", ha="center", fontsize=6, color=GRAY)
ax.text(1.0, 0.715, "photos / 2D edits", ha="center", fontsize=6, color=GRAY)
ax.set_xticks(x, [LAB[a] for a in ARMS], fontsize=6.2)
ax.set_xlim(-0.5, 4.5); ax.set_ylim(0.42, 0.73)
ax.set_ylabel("macro-F1 on 4,202 real SAGE photos")
ax.grid(axis="y", color="#e3e3e3", lw=0.5); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="lower left", fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/stage0_real.pdf"); fig.savefig(f"{OUT}/stage0_real.png", dpi=220)

# ---- macros
syn = pd.concat([pd.read_csv(f"{ROOT}/data/synthetic/stage0/{m}") for m in ("manifest_3d.csv", "manifest_2d.csv")])
cls = json.load(open(f"{ROOT}/configs/datasets/stage0_classes.json"))["classes"]
M = {}
M["DnClasses"] = len(t0["classes"]); M["DnCrops"] = len({c.split("::")[0] for c in t0["classes"]})
M["DnTest"] = f"{t0['n_real_test']:,}".replace(",", "{,}")
M["DnAssets"] = syn[syn.arm == "U3D"].asset_id.nunique()
M["DnTrain"] = f"{t0['arms']['PHOTO']['n_train']:,}".replace(",", "{,}")
M["DnGeomCls"] = len({c["class_id"] for c in cls if c.get("modality") == "geometry"})
specs = json.load(open(f"{ROOT}/data/synthetic/stage0/kb_specs.json"))
M["DnOpCls"] = sum(any(o.get("implemented") for o in v["geometry"]["operators"]) for v in specs.values())
tag = {"PHOTO": "Photo", "KB2D": "KBtwo", "GEN2D": "Gentwo", "KB3D": "KBthree", "U3D": "Uthree"}
for a in ARMS:
    r = t0["arms"][a]["real"]; M[f"DnTz{tag[a]}"] = f"{r['macro_f1']:.3f}"
    M[f"DnTz{tag[a]}Ci"] = f"[{r['macro_f1_ci'][0]:.3f}, {r['macro_f1_ci'][1]:.3f}]"
    M[f"DnTz{tag[a]}Crop"] = f"{t0['arms'][a]['real_crop']['crop_acc']:.2f}"
    M[f"DnTz{tag[a]}Iid"] = f"{t0['arms'][a]['iid']['macro_f1']:.2f}"
    r3 = t3[(a, "real")]["metrics"]; M[f"DnTt{tag[a]}"] = f"{t3m[a]:.3f}"
    sv = [t3s[(a, "real", kk)]["metrics"]["macro_f1"] for kk in SEEDS]
    M[f"DnTt{tag[a]}Sd"] = f"{np.std(sv, ddof=1):.3f}" if len(SEEDS) > 1 else "--"
    M[f"DnTt{tag[a]}Ci"] = f"[{r3['macro_f1_ci'][0]:.3f}, {r3['macro_f1_ci'][1]:.3f}]"


def sgn(v):
    return f"{v:+.3f}".replace("-", "$-$")


def pc(name, src, key):
    d = src[key]; M[name] = sgn(d["diff"]); M[name + "Ci"] = f"[{sgn(d['ci'][0])}, {sgn(d['ci'][1])}]"


pz = t0["paired_real_macro_f1"]
pc("DnPzKBthreeU", pz, "KB3D-U3D"); pc("DnPzKBthreePhoto", pz, "KB3D-PHOTO"); pc("DnPzKBtwoPhoto", pz, "KB2D-PHOTO")
pc("DnPzGenKBtwo", pz, "GEN2D-KB2D"); pc("DnPzKBtwoKBthree", pz, "KB2D-KB3D")
for nm, key in [("DnPtKBthreeU", "KB3D-U3D"), ("DnPtKBthreePhoto", "KB3D-PHOTO"), ("DnPtKBtwoPhoto", "KB2D-PHOTO"),
                ("DnPtGenKBtwo", "GEN2D-KB2D"), ("DnPtKBtwoKBthree", "KB2D-KB3D")]:
    v = t3p[key]; M[nm] = sgn(v["mean_diff"]); M[nm + "Rng"] = f"[{sgn(v['min_diff'])}, {sgn(v['max_diff'])}]"
    M[nm + "Sig"] = f"{v['seeds_ci_excl0_same_sign']}/{len(SEEDS)}"
M["DnSeeds"] = len(SEEDS)
M["DnEpochs"] = t3[("PHOTO", "real")]["epochs"]
M["DnFtMin"] = f"{np.mean([t3[(a, e)]['train_sec'] for a in ARMS for e in ('real', 'fold0')]) / 60:.0f}"
with open(f"{OUT}/diseasenet_numbers.tex", "w") as f:
    f.write("% generated by DiseaseNet/scripts/deck_stage0.py from results/E0 -- do not edit\n")
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(SEEDS, {k: (round(v['mean_diff'], 3), v['seeds_ci_excl0_same_sign']) for k, v in t3p.items()}); print(len(M), "macros")
