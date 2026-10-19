"""Figure 4 - Knowledge ablation on one specimen. Row 1 (2D, input camera, matched crop): A real, B generic, C full KB
(interim mixed-stage), C-NG KB without geometry, C-MM mismatched KB, C-stage stage-consistent KB. Row 2 (3D, rig view 0):
D no edit, E generic GE, F KB GE, K KB appearance (geo frozen), J KB geometry, H KB geometry+appearance.
Each panel: detected KB colour terms (Lab detectors) / geometric recall; required terms from the selected KB stage."""
import os, sys, json
import numpy as np, matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures.common import ROOT, TW, img, show, save, crop_white, SYN
from rendering.rig_render import load_view
from metrics.appearance_kb import main2d, required
from metrics.kb_grounding import score as kb_geo


def main(sid="Peach__Leaf_Curl__0087", key="Peach__Leaf_Curl"):
    I = f"{ROOT}/data/inputs/{sid}"; Pp = f"{ROOT}/results/pilot/{sid}"; P = f"{ROOT}/results/phaseA/{sid}"
    m = np.asarray(Image.open(f"{I}/0.png").convert("L")) > 127; ys, xs = np.nonzero(m); H, W = m.shape
    py, px = int(0.3 * (ys.max() - ys.min())), int(0.3 * (xs.max() - xs.min()))
    cb = lambda a: None if a is None else a[max(0, ys.min() - py):min(H, ys.max() + py), max(0, xs.min() - px):min(W, xs.max() + px)]
    req = sorted(required(key)[0])
    r1 = [("A: real", f"{I}/image.png"), ("B: generic", f"{Pp}/B/seed0.png"), ("C: KB, mixed stages", f"{Pp}/C/seed0.png"),
          ("C-NG: KB, no geom.", f"{Pp}/C_nogeo/seed0.png"), ("C-MM: wrong KB", f"{Pp}/C_mismatch/seed0.png"), ("C-stage: KB stage 0", f"{P}/C_stage/seed0.png")]
    r2 = [("D: no edit", f"{P}/D_render"), ("E: GE generic", f"{P}/E/seed0/render"), ("F: GE + KB", f"{P}/F/seed0/render"),
          ("K: KB appearance", f"{P}/K/seed0/render"), ("J: KB geometry", f"{P}/J/seed0/render"), ("H: J + K", f"{P}/H/seed0/render")]
    fig, axs = plt.subplots(2, 6, figsize=(TW, TW * 0.42))
    for ax, (t, p) in zip(axs[0], r1):
        a = img(p); note = ""
        if a is not None and not t.startswith("A:"):
            d = main2d(f"{I}/image.png", f"{I}/0.png", p, key); note = "detected: " + (", ".join(d["detected"]) or "none")
        show(ax, cb(a), t); ax.text(0.02, -0.04, note, transform=ax.transAxes, va="top", fontsize=4.2)
    for ax, (t, d) in zip(axs[1], r2):
        ok = os.path.exists(f"{d}/rgb/000.png"); note = ""
        if ok and not t.startswith("D:"):
            ak = f"{os.path.dirname(d)}/appearance_kb.json"; ga = f"{os.path.dirname(d)}/geom_attributes.json"
            if os.path.exists(ak): note += "colour: " + (", ".join(json.load(open(ak))["detected"]) or "none")
            if os.path.exists(ga):
                g = kb_geo(json.load(open(ga)), key); note += f"\ngeo recall {g['geometry_recall']:.2f}" if g["geometry_recall"] is not None else ""
        show(ax, crop_white(load_view(d, 0)[0]) if ok else None, t); ax.text(0.02, -0.04, note, transform=ax.transAxes, va="top", fontsize=4.2)
        if ok: ax.text(0.98, 0.98, "SYNTHETIC", transform=ax.transAxes, ha="right", va="top", **SYN)
    fig.suptitle(f"{sid.replace('__', ' ')} - KB stage-0 required colour terms: {', '.join(req) or 'none'}", fontsize=5.5, y=1.0)
    fig.subplots_adjust(hspace=0.45)
    save(fig, f"fig4_kb_ablation_{sid}", [f"results/pilot/{sid}", f"results/phaseA/{sid}"],
         "Knowledge ablation. Colour terms are detected with a priori CIELAB thresholds (5 units); geometric recall uses the synthetic-calibrated "
         "detectors. Only prompts that contain 'reddish or purple' produce a redder leaf in 2D. Renders are synthetic.")


if __name__ == "__main__":
    main(*sys.argv[1:3])
