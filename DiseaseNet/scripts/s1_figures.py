"""Stage 1 figures from result files only (DEV numbers; CPU). -> figures/stage1/*.pdf|png and figures/stage1/s1_numbers.tex
usage: ~/Desktop/Research/.venv/bin/python scripts/s1_figures.py
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R1 = f"{ROOT}/results/stage1"; OUT = f"{ROOT}/figures/stage1"
os.makedirs(OUT, exist_ok=True)
CARD, GRAY, BLUE, GOLD, INK = "#C8102E", "#707372", "#006BA6", "#C99A2E", "#333333"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.titlesize": 7.2})
M = {}


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x" if ax.get_xlim()[1] <= 1.01 and ax.get_xlim()[0] >= 0 else "y", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout(); fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png", dpi=220); plt.close(fig)


def reps(sub, name):
    v = []
    for r in range(3):
        f = f"{R1}/{sub}/{name}{'' if r == 0 else f'~{r}'}.json"
        if os.path.exists(f):
            v.append(json.load(open(f))["dev"]["macro_f1"])
    return v


# 1. generator arms
ARMS = [("PHOTO2D", "real photo (2D)", "2D"), ("GEN2D", "generic edit (2D)", "2D"), ("KB2D", "KB edit (2D)", "2D"),
        ("TEX3D", "photo texture (3D)", "3D"), ("TEXGEO3D", "photo tex. + KB ops (3D)\n= Stage-0 'KB3D'", "3D"),
        ("U3D", "unedited (3D)", "3D"), ("HARM3D", "colour-matched (3D)", "3D"), ("GEO3D", "KB ops only (3D)", "3D"),
        ("GEN3D", "generic edit (3D)", "3D"), ("KBAPP3D", "KB colour (3D)", "3D"), ("KB3D", "KB colour + ops (3D)", "3D")]
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for i, (a, lab, kind) in enumerate(ARMS):
    v = reps("stage0r", a); y = len(ARMS) - 1 - i
    ax.scatter(v, [y] * len(v), s=9, color=CARD if kind == "3D" else BLUE, alpha=0.55, lw=0, zorder=3)
    ax.scatter([np.mean(v)], [y], s=30, marker="|", color=INK, zorder=4, linewidths=1.6)
    ax.text(np.mean(v) + 0.012, y, f"{np.mean(v):.3f}", va="center", fontsize=5.8, color=INK)
    M["Arm" + a.replace("2D", "two").replace("3D", "three")] = f"{np.mean(v):.3f}"
ax.set_yticks(range(len(ARMS)), [lab for _, lab, _ in ARMS][::-1], fontsize=6)
ax.set_xlabel("DEV macro-F1 (20 classes; dots = 3 replicates, bar = mean)"); ax.set_xlim(0.3, 0.72); clean(ax)
ax.set_title("Same 60 parents, 1,920 images per arm, frozen DINOv2 probe", loc="left")
save(fig, "generator_arms")

# 2. parents vs renders at fixed images
def div(name):
    f = f"{R1}/diversity/{name}.json"
    return json.load(open(f))["dev"]["macro_f1"] if os.path.exists(f) else None
fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.3), sharey=True)
for arm, col in [("PHOTO2D", BLUE), ("KB2D", CARD)]:
    xs, ms = [], []
    for p in (3, 6, 12):
        v = [div(f"DIV:{arm}:P{p}:d{d}") for d in range(3)]; v = [x for x in v if x is not None]
        if v:
            ax[0].scatter([p] * len(v), v, s=10, color=col, alpha=0.5, lw=0); xs.append(p); ms.append(np.mean(v))
    if ms:
        ax[0].plot(xs, ms, color=col, lw=1.4, marker="o", ms=3.5, label={"PHOTO2D": "real photos", "KB2D": "KB 2D edits"}[arm])
        for x, m in zip(xs, ms):
            M[f"Div{arm.replace('2D', 'two')}P{['', '', '', 'three', '', '', 'six', '', '', '', '', '', 'twelve'][x]}"] = f"{m:.3f}"
    xs, ms = [], []
    for v_, n in ((2, 8), (4, 16), (8, 32)):
        v = [div(f"DIVR:{arm}:P3:v{v_}:d{d}") if v_ != 8 else div(f"DIV:{arm}:P3:d{d}") for d in range(3)]; v = [x for x in v if x is not None]
        if v:
            ax[1].scatter([n] * len(v), v, s=10, color=col, alpha=0.5, lw=0); xs.append(n); ms.append(np.mean(v))
    if ms:
        ax[1].plot(xs, ms, color=col, lw=1.4, marker="o", ms=3.5)
        for x, m in zip(xs, ms):
            M[f"Rend{arm.replace('2D', 'two')}N{ {8: 'eight', 16: 'sixteen', 32: 'thirtytwo'}[x]}"] = f"{m:.3f}"
ax[0].set_xscale("log", base=2); ax[0].set_xticks([3, 6, 12], ["3", "6", "12"]); ax[0].set_xlabel("independent parents / class\n(96 images / class fixed)")
ax[1].set_xscale("log", base=2); ax[1].set_xticks([8, 16, 32], ["8", "16", "32"]); ax[1].set_xlabel("images / parent\n(3 parents / class fixed)")
ax[0].set_ylabel("DEV macro-F1 (18 classes)"); ax[0].legend(frameon=False, fontsize=6)
ax[0].set_title("More parents", loc="left"); ax[1].set_title("More renders", loc="left")
for a in ax:
    clean(a)
save(fig, "parents_vs_renders")

# 3. background
B = {}
for f in glob.glob(f"{R1}/background/*.json"):
    n = os.path.basename(f)[:-5].split("~")[0]; B.setdefault(n, []).append(json.load(open(f))["dev"]["macro_f1"])
fig, ax = plt.subplots(figsize=(4.4, 2.2))
conds = [("R0", "white"), ("R1", "own\n(source-like)"), ("R2", "natural plates\n(other classes)"), ("R5", "R2 +\nphotometric")]
for a, col, lab in [("PHOTO2D", BLUE, "real photo"), ("TEX3D", GOLD, "photo-textured 3D"), ("U3D", CARD, "unedited 3D")]:
    m = [np.mean(B[f"{a}@{c}"]) for c, _ in conds]
    ax.plot(range(4), m, color=col, lw=1.3, marker="o", ms=3.5, label=lab)
ax.axhline(np.mean(B["PLATE2D@R1"]), color=GRAY, ls="--", lw=0.8)
ax.text(0, np.mean(B["PLATE2D@R1"]) + 0.012, f"background only {np.mean(B['PLATE2D@R1']):.2f} (chance 0.05)", fontsize=5.8, color=GRAY)
ax.set_xticks(range(4), [l for _, l in conds], fontsize=6); ax.set_xlim(-0.3, 3.3); ax.set_ylabel("DEV macro-F1")
ax.legend(frameon=False, fontsize=5.8, loc="lower right", bbox_to_anchor=(1.0, 0.12))
ax.set_title("Background condition (8 images / parent)", loc="left"); clean(ax)
save(fig, "background")
M["BgOnly"] = f"{np.mean(B['PLATE2D@R1']):.3f}"

# 4. modality x generator
C = pd.read_csv(f"{R1}/stage0r/contrasts_contrasts_stage0r_dev.csv").set_index("name")
rows = [("U3D − PHOTO2D", "geometry cohort: U3D - PHOTO2D", "appearance cohort: U3D - PHOTO2D"),
        ("KB3D − KB2D", "geometry cohort: KB3D - KB2D", "appearance cohort: KB3D - KB2D"),
        ("KB ops on KB colour", "geometry cohort: KB3D - KBAPP3D", "appearance cohort: KB3D - KBAPP3D")]
fig, ax = plt.subplots(figsize=(4.4, 1.9))
for i, (lab, g, a) in enumerate(rows):
    y = 2 - i
    for k, col, off in ((g, CARD, 0.12), (a, GRAY, -0.12)):
        r = C.loc[k]; ax.errorbar(r["diff"], y + off, xerr=[[r["diff"] - r.ci_lo], [r.ci_hi - r["diff"]]], fmt="o", ms=3.5, color=col, lw=1, capsize=1.5)
ax.axvline(0, color=INK, lw=0.6); ax.set_yticks(range(3), [r[0] for r in rows][::-1], fontsize=6.2)
ax.scatter([], [], color=CARD, s=12, label="geometry cohort (4 classes)"); ax.scatter([], [], color=GRAY, s=12, label="appearance cohort (16)")
ax.legend(frameon=False, fontsize=5.8, loc="lower left"); ax.set_xlabel("Δ DEV macro-F1 within cohort (3 replicates, 95% CI)")
ax.set_title("3D helps only where the symptom is a shape", loc="left"); clean(ax)
save(fig, "modality_interaction")
for k, n in [("geometry cohort: KB3D - KB2D", "GeoKBthreeKBtwo"), ("appearance cohort: KB3D - KB2D", "AppKBthreeKBtwo"),
             ("geometry cohort: U3D - PHOTO2D", "GeoUPhoto"), ("appearance cohort: U3D - PHOTO2D", "AppUPhoto"),
             ("P1 KB3D - U3D", "PoneKBU"), ("P2 KB3D - KB2D", "PtwoKBKB"), ("P3 MIX50 - KB2D", "PthreeMix"),
             ("texture: TEX3D - U3D", "TexU"), ("harmonise: HARM3D - U3D", "HarmU"), ("KB in 2D: KB2D - GEN2D", "KBGenTwo"),
             ("KB in 3D: KBAPP3D - GEN3D", "KBGenThree"), ("render gap: TEX3D - PHOTO2D", "RenderGap"),
             ("geometry cohort: KB3D - KBAPP3D", "GeoOps")]:
    r = C.loc[k]; M[f"C{n}"] = f"{r['diff']:+.3f}".replace("-", "$-$"); M[f"C{n}Ci"] = f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]".replace("-", "$-$")
for a, b, n in [("KB3D", "KB2D", "IntKB"), ("U3D", "PHOTO2D", "IntU")]:
    j = json.load(open(f"{R1}/stage0r/interaction_{a}_{b}.json"))
    M[n] = f"{j['interaction']:+.3f}"; M[n + "Ci"] = f"[{j['ci'][0]:+.3f}, {j['ci'][1]:+.3f}]"

# 5. knowledge-only test
W = pd.read_csv(f"{R1}/cross/within_crop.csv").set_index("set")
order = [("XPHOTO2D", "sibling photo, relabelled"), ("XGEN2D", "+ generic edit"), ("XKBAPP3D", "+ target KB colour (3D)"),
         ("XKB3D", "+ target KB colour + ops (3D)"), ("XKB2D", "+ target KB edit (2D)")]
fig, ax = plt.subplots(figsize=(4.2, 1.8))
for i, (s, lab) in enumerate(order):
    r = W.loc[s]; y = len(order) - 1 - i
    ax.errorbar(r.within_crop_acc, y, xerr=[[r.within_crop_acc - r.lo], [r.hi - r.within_crop_acc]], fmt="o", ms=3.5,
                color=CARD if "KB" in s else GRAY, lw=1, capsize=1.5)
    ax.text(r.hi + 0.01, y, f"{r.within_crop_acc:.2f}", va="center", fontsize=5.8)
    M[s.replace('2D', 'two').replace('3D', 'three')] = f"{r.within_crop_acc:.3f}"
ax.axvline(0.5, color=INK, lw=0.6, ls="--"); ax.text(0.505, 4.3, "chance", fontsize=5.6, color=INK)
ax.set_yticks(range(len(order)), [l for _, l in order][::-1], fontsize=6); ax.set_xlim(0.15, 0.6)
ax.set_xlabel("within-crop disease discrimination on DEV (16 classes)")
ax.set_title("Knowledge-only test (parents = the crop's other disease)", loc="left", fontsize=6.8); clean(ax)
save(fig, "knowledge_only")

# 6. few-shot
fig, ax = plt.subplots(figsize=(3.6, 2.4))
for s, col, lab in [("REAL", INK, "real shots only"), ("REAL+3", GRAY, "real shots + 3 parents"), ("PHOTO2D", BLUE, "PHOTO2D + shots"),
                    ("U3D", CARD, "U3D + shots"), ("KB2D", GOLD, "KB2D + shots")]:
    f = f"{R1}/fewshot/{s}.json"
    if not os.path.exists(f):
        continue
    d = json.load(open(f))["dev_macro_f1"]; ks = sorted(int(k) for k in d); m = [np.mean(d[str(k)]) for k in ks]
    ax.plot([k + 0.5 for k in ks], m, marker="o", ms=3, lw=1.2, color=col, label=lab)
    M[f"Fs{s.replace('+3', 'pthree').replace('2D', 'two').replace('3D', 'three')}Sixteen"] = f"{np.mean(d['16']):.3f}"
ax.set_xscale("log", base=2); ax.set_xticks([0.5, 1.5, 4.5, 8.5, 16.5], ["0", "1", "4", "8", "16"]); ax.set_xlabel("real shots per class (adapt partition)")
ax.set_ylabel("DEV macro-F1"); ax.legend(frameon=False, fontsize=5.6); ax.set_title("Adapted S1 (frozen probe)", loc="left"); clean(ax)
save(fig, "fewshot")

# 7. feature preservation
FP = pd.read_csv(f"{R1}/feature_preservation/summary.csv").set_index("arm")
fig, ax = plt.subplots(figsize=(3.6, 1.8))
arms = [a for a in ["U3D", "HARM3D", "GEN3D", "KBAPP3D", "TEX3D"] if a in FP.index]
ax.barh(range(len(arms)), FP.loc[arms, "hf_ratio"], color=[CARD if a == "TEX3D" else GRAY for a in arms], height=0.55)
for i, a in enumerate(arms):
    ax.text(FP.loc[a, "hf_ratio"] + 0.01, i, f"{FP.loc[a, 'hf_ratio']:.2f}", va="center", fontsize=5.8)
ax.axvline(1, color=INK, lw=0.6, ls="--"); ax.text(0.98, len(arms) - 0.6, "photo", ha="right", fontsize=5.8)
ax.set_yticks(range(len(arms)), arms, fontsize=6); ax.set_xlim(0, 1.05); ax.set_xlabel("fine-texture energy kept at the input view (render / photo)")
ax.set_title("3D assets lose most lesion-scale texture", loc="left"); clean(ax)
save(fig, "feature_preservation")
M["HFU"] = f"{FP.loc['U3D', 'hf_ratio']:.2f}"; M["HFTEX"] = f"{FP.loc['TEX3D', 'hf_ratio']:.2f}"

with open(f"{OUT}/s1_numbers.tex", "w") as f:
    f.write("% generated by DiseaseNet/scripts/s1_figures.py from results/stage1 -- do not edit\n")
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(len(M), "macros")
