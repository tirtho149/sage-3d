"""Figure 3 - Geometry editing: arms D/E/F/G/J/K/H of one specimen on fixed rig views, matched crops per view,
with per-arm measured geometry recall, coherence and grain. All renders synthetic and unverified."""
import os, sys, json
import numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures.common import ROOT, TW, show, save, SYN
from rendering.rig_render import load_view
from metrics.kb_grounding import score as kb_geo

LAB = {"D": "D unedited", "E": "E GE generic", "F": "F GE + KB (orig.)", "G": "G generic geometry", "J": "J KB geometry",
       "K": "K KB appearance (GE, geo frozen)", "H": "H = J then K"}


def main(sid="Peach__Leaf_Curl__0087", key="Peach__Leaf_Curl", views=(0, 2, 4, 6, 16, 17), arms=("D", "E", "F", "G", "J", "K", "H")):
    P = f"{ROOT}/results/phaseA/{sid}"
    dirs = {a: (f"{P}/D_render" if a == "D" else f"{P}/{a}/seed0/render") for a in arms}
    dirs = {a: d for a, d in dirs.items() if os.path.exists(f"{d}/rgb/000.png")}
    boxes = {}
    for v in views:                                     # matched crop per view = union of silhouettes over arms
        m = np.zeros((512, 512), bool)
        for d in dirs.values(): m |= load_view(d, v)[1] > 0.05
        ys, xs = np.nonzero(m); cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        h = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 + 8                     # square crop -> equal panel sizes
        y0, x0 = int(max(0, cy - h)), int(max(0, cx - h)); boxes[v] = (y0, int(min(512, y0 + 2 * h)), x0, int(min(512, x0 + 2 * h)))
    fig, axs = plt.subplots(len(dirs), len(views) + 1, figsize=(TW, TW * len(dirs) / (len(views) + 1) * 0.72), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1] * len(views) + [1.25]))
    for r, (a, d) in enumerate(dirs.items()):
        for c, v in enumerate(views):
            y0, y1, x0, x1 = boxes[v]; show(axs[r][c], load_view(d, v)[0][y0:y1, x0:x1], (f"view {v}" if r == 0 else None))
        ax = axs[r][-1]; ax.axis("off"); txt = LAB.get(a, a)
        ga = f"{P}/{a}/seed0/geom_attributes.json"; gd = f"{P}/{a}/seed0/gaussian_diff.json"; vm = f"{P}/{a}/seed0/view_metrics.json"
        if os.path.exists(ga):
            g = kb_geo(json.load(open(ga)), key); txt += f"\ngeo recall {g['geometry_recall']:.2f}" if g["geometry_recall"] is not None else "\ngeo recall n/a"
        if os.path.exists(gd):
            g_ = json.load(open(gd)); txt += "\ncoherence n/a (no motion)" if g_["disp_rel"]["p95"] < 1e-6 else f"\ncoherence {g_['disp_coherence_k16']['median']:.2f}"
        if os.path.exists(vm): txt += f"\ngrain x{json.load(open(vm))['summary']['grain_ratio']:.1f}"
        ax.text(0.02, 0.5, txt, va="center", fontsize=4.6, transform=ax.transAxes)
        ax.text(0.02, 0.06, "SYNTHETIC, unverified", transform=ax.transAxes, **SYN)
    save(fig, f"fig3_editing_{sid}", [f"results/phaseA/{sid}/*/seed0/render", f"results/phaseA/{sid}/*/seed0/*.json"],
         "Same specimen, fixed rig views (matched crops per column). Geometry recall = fraction of the selected KB stage's geometric "
         "attributes detected (specific detectors, synthetic-calibrated thresholds). Coherence: 1 = smooth deformation, 0.25 = i.i.d. jitter. "
         "Every view is a synthetic render of a synthetic asset; novel views are unverified.")


if __name__ == "__main__":
    main(*sys.argv[1:3])
