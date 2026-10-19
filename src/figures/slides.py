"""Slide-sized figures for the advisor deck, generated from result files only (no hand-entered numbers).

Output: figures/slides/*.pdf (vector charts) and *.png (300 dpi image panels), plus slides_numbers.json
holding every number quoted on a slide, so the deck text can be checked against results.
Usage: python src/figures/slides.py
"""
import os, json, csv, glob, time
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/figures/slides"; os.makedirs(OUT, exist_ok=True)
R = f"{ROOT}/results"
# Figures are drawn at the PHYSICAL size of their slide box (mm / 25.4), so 6.5-7 pt here is 6.5-7 pt on the slide.
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "axes.linewidth": 0.6,
                     "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6.3, "pdf.fonttype": 42, "lines.linewidth": 1.0,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "axes.spines.top": False, "axes.spines.right": False})
CARD, BLUE, GOLD, GRAY = "#C8102E", "#006BA6", "#F1BE48", "#707372"
NUM = {}


def img(p):
    return np.asarray(Image.open(p).convert("RGB")) if p and os.path.exists(p) else None


def crop_white(a, pad=8, thr=248):
    m = (a < thr).any(-1); ys, xs = np.nonzero(m)
    return a if len(ys) == 0 else a[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]


def show(ax, a, title=None, tag=None, tagc=CARD):
    ax.axis("off")
    if a is None:
        ax.text(.5, .5, "not available", ha="center", va="center", color=GRAY, transform=ax.transAxes)
    else:
        ax.imshow(a, interpolation="lanczos")
    if title: ax.set_title(title, pad=2, fontsize=6.3)
    if tag: ax.text(0.5, -0.04, tag, transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=tagc, fontweight="bold")


def save(fig, name, vector):
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    if vector: fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def short(sid):
    c, d, n = sid.split("__")
    return f"{c} {n}"


# ---------------------------------------------------------------- E1: alignment chart (all 15) + 4-specimen panel
al = json.load(open(f"{R}/alignment/summary.json"))
al = sorted(al, key=lambda r: r["A0_iou"])
fig, ax = plt.subplots(figsize=(2.8, 2.35))
y = np.arange(len(al))
for i, r in enumerate(al):
    ax.plot([r["A0_iou"], r["A3s_iou"]], [i, i], color="#bbbbbb", lw=1.0, zorder=1)
ax.scatter([r["A0_iou"] for r in al], y, color=GRAY, label="A0: SAM 3D pose", zorder=2, s=10)
ax.scatter([r["A3s_iou"] for r in al], y, color=CARD, label="A3*: pose-only fit", zorder=3, s=10)
for i, r in enumerate(al):
    if r["label"] != "ok":
        ax.text(r["A3s_iou"] + 0.015, i, "recon. failure", va="center", fontsize=5.8, color=CARD)
ax.set_yticks(y); ax.set_yticklabels([short(r["specimen"]) for r in al])
ax.set_xlabel("input-view mask IoU (fitted, not held-out)"); ax.set_xlim(0.35, 1.0)
ax.legend(loc="upper left", frameon=False)
save(fig, "e1_iou", True)
a0, a3 = np.array([r["A0_iou"] for r in al]), np.array([r["A3s_iou"] for r in al])
NUM["E1"] = dict(n=len(al), median_A0=round(float(np.median(a0)), 2), median_A3s=round(float(np.median(a3)), 2),
                 improved=int((a3 > a0).sum()), oversized=int(sum(r["A0_area_err"] > 0 for r in al)),
                 depth_worse=int(sum(r["A3s_depth_rel_err"] > r["A0_depth_rel_err"] + 0.01 for r in al)),
                 lpips_worse=int(sum(r["A3s_lpips"] - r["A0_lpips"] > 0.02 for r in al)))

sel = ["Tomato__Leaf_Mold__0015", "Corn__Smut__0012", "Peach__Leaf_Curl__0087", "Peach__Leaf_Curl__0006"]
fig, axs = plt.subplots(3, 4, figsize=(3.4, 2.25), gridspec_kw=dict(hspace=0.3, wspace=0.08))
byid = {r["specimen"]: r for r in al}
for j, s in enumerate(sel):
    r = byid[s]; d = f"{R}/alignment/{s}"
    show(axs[0, j], img(f"{ROOT}/data/inputs/{s}/image.png"), short(s))
    show(axs[1, j], img(f"{d}/overlay_A0.png"), f"A0  IoU {r['A0_iou']:.2f}")
    ov = f"{d}/overlay_A3_render_compare.png" if r["selected"].startswith("A3") else f"{d}/overlay_A0.png"
    show(axs[2, j], img(ov) if r["selected"].startswith("A3") else img(f"{d}/A2_closed_form.png"),
         f"A3*  IoU {r['A3s_iou']:.2f}" + ("  (A2 kept)" if not r["selected"].startswith("A3") else ""),
         "RECON. FAILURE" if r["label"] != "ok" else None)
save(fig, "e1_panel", False)

# ---------------------------------------------------------------- 3D arms on Peach 0087 (views 0, 2, 16)
s = "Peach__Leaf_Curl__0087"; P = f"{R}/phaseA/{s}"
arms = [("D", f"{P}/D_render/rgb", "D  unedited"), ("E", f"{P}/E/seed0/render/rgb", "E  GaussianEditor, generic"),
        ("F", f"{P}/F/seed0/render/rgb", "F  GaussianEditor + KB"), ("G", f"{P}/G/seed0/render/rgb", "G  generic geometry"),
        ("J", f"{P}/J/seed0/render/rgb", "J  KB geometry"), ("K", f"{P}/K/seed0/render/rgb", "K  KB colour, geometry frozen"),
        ("H", f"{P}/H/seed0/render/rgb", "H  J then K")]
views = [0, 2, 16]
fig, axs = plt.subplots(len(views), len(arms), figsize=(5.6, 1.85), gridspec_kw=dict(hspace=0.04, wspace=0.03))
for j, (a, d, t) in enumerate(arms):
    for i, v in enumerate(views):
        im = img(f"{d}/{v:03d}.png")
        show(axs[i, j], crop_white(im) if im is not None else None, t.split("  ")[0] if i == 0 else None)
for i, v in enumerate(views):
    b = axs[i, 0].get_subplotspec().get_position(fig)
    fig.text(b.x0 - 0.004, (b.y0 + b.y1) / 2, f"view {v}", rotation=90, ha="right", va="center", fontsize=6)
fig.text(0.5, 0.0, "All panels SYNTHETIC renders on the same fixed rig; nothing here is verified against a real view.",
         ha="center", fontsize=5.8, color=CARD)
save(fig, "arms_0087", False)

# ---------------------------------------------------------------- quantitative 3D comparisons (specimen = unit)
rows = list(csv.DictReader(open(f"{R}/statistics/specimen_metrics.csv")))
def col(arm, key):
    return {r["sid"]: float(r[key]) for r in rows if r["arm"] == arm and r[key] not in ("", "nan")}

fig, axs = plt.subplots(1, 3, figsize=(3.3, 1.75), gridspec_kw=dict(wspace=0.62))
# (a) KB geometric recall: J vs G paired
g, j = col("G", "geometry_recall"), col("J", "geometry_recall"); ks = sorted(set(g) & set(j))
ax = axs[0]
for k in ks:
    ax.plot([0, 1], [g[k], j[k]], color="#cccccc", lw=1.2, zorder=1)
ax.scatter([0] * len(ks), [g[k] for k in ks], color=GRAY, zorder=2, s=8); ax.scatter([1] * len(ks), [j[k] for k in ks], color=CARD, zorder=2, s=8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["G", "J"]); ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.05, 1.05)
ax.set_ylabel("KB geom. recall"); ax.set_title(f"(a) recall, n={len(ks)}")
# (b) displacement coherence
ax = axs[1]; arms3 = ["E", "F", "G", "J"]
vals = [list(col(a, "coherence").values()) for a in arms3]
for i, v in enumerate(vals):
    ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=[GOLD, GOLD, GRAY, CARD][i], s=6)
    ax.text(i, 1.08, f"{len(v)}", ha="center", fontsize=5.5, color=GRAY)
ax.axhline(0.25, color=GRAY, ls="--", lw=0.8); ax.text(3.45, 0.29, "noise", fontsize=5.5, color=GRAY, ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels(["E", "F", "G", "J"]); ax.set_ylim(0, 1.15)
ax.set_title("(b) coherence")
# (c) grain
ax = axs[2]; arms4 = ["E", "F", "K", "H", "G", "J"]
vals = [list(col(a, "grain_ratio").values()) for a in arms4]
for i, v in enumerate(vals):
    ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color=([GOLD] * 4 + [GRAY, CARD])[i], s=6)
    ax.text(i, 32, f"{len(v)}", ha="center", fontsize=5.5, color=GRAY)
ax.axhline(1, color=GRAY, ls="--", lw=0.8); ax.set_yscale("log"); ax.set_ylim(0.7, 45)
ax.set_xticks(range(6)); ax.set_xticklabels(arms4); ax.set_title("(c) grain vs D (log)")
save(fig, "quant3d", True)
NUM["geom_recall"] = dict(n=len(ks), G=round(np.mean([g[k] for k in ks]), 2), J=round(np.mean([j[k] for k in ks]), 2))
NUM["coherence_median"] = {a: round(float(np.median(list(col(a, "coherence").values()))), 2) for a in arms3 if col(a, "coherence")}
NUM["grain_median"] = {a: round(float(np.median(list(col(a, "grain_ratio").values()))), 1) for a in arms4 if col(a, "grain_ratio")}
NUM["n_arm"] = {a: len(col(a, "grain_ratio")) for a in ["D", "E", "F", "G", "J", "K", "H"]}

# ---------------------------------------------------------------- 2D arms: KB colour recall (all specimens available)
m2 = json.load(open(f"{R}/statistics/metrics2d_summary.json"))
order = [("B", "B generic"), ("C_nogeo", "C-NG KB, wrong stage"), ("C_mismatch", "C-MM other disease"),
         ("C", "C KB (mixed stages)"), ("C_stage", "C-stage KB stage"), ("I", "I KB, shape-capable")]
fig, ax = plt.subplots(figsize=(2.9, 1.65))
for i, (k, lab) in enumerate(order):
    v = m2[k]; ax.barh(i, v["appearance_recall"], color=CARD if k in ("C", "C_stage", "I") else GRAY, height=0.62)
    ax.text(v["appearance_recall"] + 0.02, i, f"{v['appearance_recall']:.2f} (n={v['n_specimens']})", va="center", fontsize=5.8)
ax.set_yticks(range(len(order))); ax.set_yticklabels([l for _, l in order]); ax.set_xlim(0, 1.15)
ax.set_xlabel("KB colour recall (specimen mean)")
save(fig, "quant2d", True)
NUM["2d"] = {k: dict(recall=round(m2[k]["appearance_recall"], 2), n=m2[k]["n_specimens"],
                     outside_change=round(m2[k]["outside_change"], 2)) for k, _ in order}

# ---------------------------------------------------------------- cross-view consistency
cv = {}
for f in glob.glob(f"{R}/statistics/crossview_corr_*.json"):
    for arm, v in json.load(open(f)).items():
        a = arm.split(" ")[0]
        if v and v.get("median_dE_adjacent") is not None:
            cv.setdefault(a, []).append(v["median_dE_adjacent"])
order = [a for a in ["D", "G", "J", "E", "K", "H", "F", "Bmv", "Cmv"] if a in cv]
fig, ax = plt.subplots(figsize=(2.65, 1.5))
for i, a in enumerate(order):
    v = cv[a]; c = CARD if a in ("Bmv", "Cmv") else (GOLD if a in ("E", "F", "K", "H") else GRAY)
    ax.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, color=c, s=5)
    ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=0.9)
    ax.text(i, 1.05, f"{len(v)}", ha="center", fontsize=5.5, color=GRAY, transform=ax.get_xaxis_transform())
labels = {"D": "D\nfloor", "Bmv": "Bmv\n2D", "Cmv": "Cmv\n2D+KB"}
ax.set_xticks(range(len(order))); ax.set_xticklabels([labels.get(a, a) for a in order])
ax.set_ylabel("same-point ΔE\n(lower = consistent)")
save(fig, "crossview", True)
NUM["crossview_median"] = {a: dict(median=round(float(np.median(cv[a])), 1), n=len(cv[a])) for a in order}

# ---------------------------------------------------------------- 2D KB strip (Peach 0087, input view): replaces the paper composite
s2 = "Peach__Leaf_Curl__0087"; I2 = f"{ROOT}/data/inputs/{s2}"
msk = np.asarray(Image.open(f"{I2}/0.png").convert("L")) > 127; yy, xx = np.nonzero(msk); H2, W2 = msk.shape
py2, px2 = int(0.25 * np.ptp(yy)), int(0.25 * np.ptp(xx))
cb2 = lambda a: None if a is None else a[max(0, yy.min() - py2):min(H2, yy.max() + py2), max(0, xx.min() - px2):min(W2, xx.max() + px2)]
def rs(p):
    a = img(p); return None if a is None else np.asarray(Image.fromarray(a).resize((W2, H2)))
strip = [(img(f"{I2}/image.png"), "A real", "REAL", "#2e7d32"), (rs(f"{R}/pilot/{s2}/B/seed0.png"), "B generic", "2D EDIT", "#8B5B29"),
         (rs(f"{R}/pilot/{s2}/C/seed0.png"), "C KB", "2D EDIT", "#8B5B29"), (rs(f"{R}/pilot/{s2}/C_nogeo/seed0.png"), "C-NG", "2D EDIT", "#8B5B29"),
         (rs(f"{R}/pilot/{s2}/C_mismatch/seed0.png"), "C-MM", "2D EDIT", "#8B5B29"), (rs(f"{P}/C_stage/seed0.png"), "C-stage", "2D EDIT", "#8B5B29"),
         (rs(f"{P}/I/seed0.png"), "I shape", "2D EDIT", "#8B5B29")]
fig, axs = plt.subplots(1, len(strip), figsize=(5.6, 1.15), gridspec_kw=dict(wspace=0.05))
for ax, (a, t, tag, c) in zip(axs, strip):
    show(ax, cb2(a), t, tag, c)
save(fig, "kb2d_strip", False)

# ---------------------------------------------------------------- cross-view strip: per-view 2D KB edits vs one 3D asset (Peach 0087)
rows_cv = [("Cmv", [f"{R}/pilot/{s2}/Cmv_masked/{i:03d}.png" for i in range(8)], "per-view 2D KB edits"),
           ("H", [f"{P}/H/seed0/render/rgb/{i:03d}.png" for i in range(8)], "one 3D asset (H)")]
fig, axs = plt.subplots(2, 8, figsize=(4.0, 1.25), gridspec_kw=dict(wspace=0.02, hspace=0.06))
for r, (k, ps_, lab) in enumerate(rows_cv):
    for i, p_ in enumerate(ps_):
        a = img(p_); show(axs[r, i], crop_white(a) if a is not None else None, f"view {i}" if r == 0 else None)
    b = axs[r, 0].get_subplotspec().get_position(fig)
    fig.text(b.x0 - 0.006, (b.y0 + b.y1) / 2, lab.replace(" ", "\n", 1), rotation=0, ha="right", va="center", fontsize=5.6)
save(fig, "crossview_strip", False)

# ---------------------------------------------------------------- 2x2 ablation
ab = json.load(open(f"{R}/statistics/ablation_2x2.json"))
NUM["ablation_2x2"] = ab

# ---------------------------------------------------------------- failure panel (clean titles)
def closeup(a):
    if a is None: return None
    a = crop_white(a); h, w = a.shape[:2]
    return a[int(.05 * h):int(.65 * h), int(.05 * w):int(.45 * w)]

F6 = "Peach__Leaf_Curl__0006"
pan = [(img(f"{ROOT}/data/inputs/{F6}/image.png"), "Peach 0006 photo\n(curled leaf)", "REAL", "#2e7d32"),
       (crop_white(img(f"{R}/phaseA/{F6}/D_render/rgb/000.png")), "SAM 3D:\nclosed ring", "SYNTHETIC", CARD),
       (img(f"{R}/alignment/Peach__Leaf_Curl__0063/overlay_A3_render_compare.png"), "pose fit A3 diverged\n(0063, IoU 0.00)", None, CARD),
       (closeup(img(f"{P}/E/seed0/render/rgb/000.png")),
        "GaussianEditor:\ngrain + jitter", "SYNTHETIC", CARD),
       (img(f"{P}/I/seed0.png"), "2D shape edit:\npuckering flattened", "EDITED REAL (2D)", "#8B5B29")]
fig, axs = plt.subplots(1, len(pan), figsize=(5.6, 1.45), gridspec_kw=dict(wspace=0.1))
for ax, (a, t, tag, c) in zip(axs, pan):
    show(ax, a, t, tag, c)
save(fig, "failures", False)

NUM["generated"] = time.strftime("%Y-%m-%d %H:%M")
json.dump(NUM, open(f"{OUT}/slides_numbers.json", "w"), indent=1, default=float)
print(json.dumps(NUM, indent=1, default=float))

# ---------------------------------------------------------------- LaTeX macros: every number quoted in the deck
comp = {(c["a"], c["b"], c["metric"]): c for c in json.load(open(f"{R}/statistics/comparisons.json"))}
def f2(x): return f"{x:.2f}"
M = {}
e = NUM["E1"]
M.update(EoneN=e["n"], EoneAzero=f2(e["median_A0"]), EoneAthree=f2(e["median_A3s"]), EoneImproved=e["improved"],
         EoneOversized=e["oversized"], EoneDepthWorse=e["depth_worse"], EoneLpipsWorse=e["lpips_worse"])
c = comp[("J", "G", "geometry_recall")]
M.update(GRn=c["n"], GRcls=c["n_classes"], GRj=f2(c["mean_a"]), GRg=f2(c["mean_b"]), GRd=f"{c['diff']:+.2f}",
         GRlo=f"{c['ci95_specimen_bootstrap'][0]:+.2f}", GRhi=f"{c['ci95_specimen_bootstrap'][1]:+.2f}",
         GRpos=c["n_positive"], GRneg=c["n_negative"], GRp=f"{c['wilcoxon_p']:.3f}", GRholm=f2(c["p_holm"]))
c = comp[("J", "G", "unsupported_rate")]
M.update(USn=c["n"], USj=f2(c["mean_a"]), USg=f2(c["mean_b"]), USlo=f"{c['ci95_specimen_bootstrap'][0]:+.2f}",
         UShi=f"{c['ci95_specimen_bootstrap'][1]:+.2f}")
for a, k in [("E", "CohE"), ("F", "CohF"), ("G", "CohG"), ("J", "CohJ")]:
    M[k] = f2(NUM["coherence_median"][a])
for a in ["E", "F", "K", "H", "G", "J"]:
    M[f"Grain{a}"] = f"{NUM['grain_median'][a]:.1f}"
for a in ["D", "E", "F", "G", "J", "K", "H"]:
    M[f"n{a}"] = NUM["n_arm"][a]
for k, nm in [("B", "B"), ("C", "C"), ("C_nogeo", "CNG"), ("C_mismatch", "CMM"), ("C_stage", "Cst"), ("I", "I")]:
    M[f"Rec{nm}"] = f2(NUM["2d"][k]["recall"]); M[f"RecN{nm}"] = NUM["2d"][k]["n"]
M["OutI"] = f2(NUM["2d"]["I"]["outside_change"])
for a, v in NUM["crossview_median"].items():
    M[f"Cv{a}"] = f"{v['median']:.1f}"; M[f"CvN{a}"] = v["n"]
for (a_, b_, met), nm in [(("J", "E", "coherence"), "CohJE"), (("J", "E", "grain_ratio"), "GrJE"), (("K", "E", "grain_ratio"), "GrKE"),
                          (("H", "F", "grain_ratio"), "GrHF"), (("K", "E", "fuzz_ratio"), "FzKE"), (("H", "F", "geometry_recall"), "GeoHF")]:
    c = comp.get((a_, b_, met))
    if c is None: continue
    M.update({f"{nm}n": c["n"], f"{nm}cls": c["n_classes"], f"{nm}a": f2(c["mean_a"]), f"{nm}b": f2(c["mean_b"]),
              f"{nm}lo": f"{c['ci95_specimen_bootstrap'][0]:+.2f}", f"{nm}hi": f"{c['ci95_specimen_bootstrap'][1]:+.2f}",
              f"{nm}holm": (f"{c['p_holm']:.3f}" if c.get("p_holm") is not None else "--")})
M["UScls"] = comp[("J", "G", "unsupported_rate")]["n_classes"]; M["USholm"] = f2(comp[("J", "G", "unsupported_rate")]["p_holm"])
cells = NUM["ablation_2x2"]["cells"]
M.update(AbD=f2(cells["D (no geo, no KB)"]), AbK=f2(cells["K (no geo, KB)"]), AbG=f2(cells["G (geo, no KB)"]),
         AbH=f2(cells["H (geo, KB)"]), AbInt=f"{NUM['ablation_2x2']['interaction_mean']:+.2f}",
         AbN=NUM["ablation_2x2"]["n_specimens_complete"])
# run progress, counted from the asset files on disk
pil = sorted(glob.glob(f"{R}/pilot/*__*/"))
M["RunEdone"] = sum(os.path.exists(f"{d}E/seed0/last.ply") or os.path.exists(f"{ROOT}/results/phaseA/{os.path.basename(d[:-1])}/E/seed0/asset.ply") for d in pil)
M["RunFdone"] = sum(os.path.exists(f"{d}F/seed0/last.ply") or os.path.exists(f"{ROOT}/results/phaseA/{os.path.basename(d[:-1])}/F/seed0/asset.ply") for d in pil)
M["RunKdone"] = len(glob.glob(f"{R}/phaseA/*/K/seed0/asset.ply"))
M["RunJdone"] = len({p.split("/")[-4] for p in glob.glob(f"{R}/phaseA/*/J/seed*/asset.ply")})
M["RunTwoDdone"] = len({p.split("/")[-3] for p in glob.glob(f"{R}/phaseA/*/C_stage/seed*.png")})
M["RunSpec"] = len(pil)
M["NumGenerated"] = NUM["generated"]
with open(f"{OUT}/numbers.tex", "w") as fh:
    fh.write("% generated by src/figures/slides.py from results/ -- do not edit by hand\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("macros:", len(M))
