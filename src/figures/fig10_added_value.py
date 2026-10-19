"""Figure 10 - What a KB-edited 3D asset adds beyond 2D, and what it does NOT establish.
Row 1 (input camera): real photo | 2D KB edit (C-stage) | 2D shape-capable KB edit (I) | 3D: D and J (or H) rendered at the
corrected input pose. Row 2 (views the photo does not contain): side/oblique/bottom renders of D vs J (or H).
Row 3: failure case (reconstruction failure) - real photo vs its SAM 3D asset. Labels: REAL / EDITED REAL / SYNTHETIC unverified."""
import os, sys, json
import numpy as np, matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures.common import ROOT, TW, img, show, save, crop_white, SYN
from rendering.rig_render import load_view
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
from camera_alignment.align import render_at_input


def tag(ax, t, c):
    ax.text(0.02, 0.98, t, transform=ax.transAxes, va="top", fontsize=4.5, fontweight="bold", color=c,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6))


def main(sid="Peach__Leaf_Curl__0087", fail="Peach__Leaf_Curl__0006", arm3d=None):
    P = f"{ROOT}/results/phaseA/{sid}"; I = f"{ROOT}/data/inputs/{sid}"
    arm3d = arm3d or ("H" if os.path.exists(f"{P}/H/seed0/asset.ply") else "J")
    d_in, _ = render_at_input(f"{ROOT}/results/pilot/sam3d/{sid}/seed0/splat_ge.ply", sid)
    x_in, _ = render_at_input(f"{P}/{arm3d}/seed0/asset.ply", sid)
    m = np.asarray(Image.open(f"{I}/0.png").convert("L")) > 127; ys, xs = np.nonzero(m); H, W = m.shape
    py, px = int(0.35 * (ys.max() - ys.min())), int(0.35 * (xs.max() - xs.min()))
    cb = lambda a: a[max(0, ys.min() - py):min(H, ys.max() + py), max(0, xs.min() - px):min(W, xs.max() + px)]   # matched crop, row 1
    fig, axs = plt.subplots(3, 5, figsize=(TW, TW * 0.68))
    row1 = [(cb(img(f"{I}/image.png")), "A: real photo", "REAL", "#006400"), (cb(img(f"{P}/C_stage/seed0.png")), "C-stage: 2D KB edit", "EDITED REAL (2D)", "#7f6000"),
            (cb(img(f"{P}/I/seed0.png")), "I: 2D shape-capable", "EDITED REAL (2D)", "#7f6000"),
            (cb(d_in), "D @ input cam (A3*)", "SYNTHETIC", "#b00000"), (cb(x_in), f"{arm3d} @ input cam", "SYNTHETIC", "#b00000")]
    for ax, (a, t, lab, c) in zip(axs[0], row1): show(ax, a, t); tag(ax, lab, c)
    views = [(2, "side (az 90)"), (6, "side (az 270)"), (17, "bottom")]
    for k, (v, name) in enumerate(views[:2]):
        show(axs[1][2 * k], crop_white(load_view(f"{P}/D_render", v)[0]), f"D: {name}"); tag(axs[1][2 * k], "SYNTHETIC", "#b00000")
        show(axs[1][2 * k + 1], crop_white(load_view(f"{P}/{arm3d}/seed0/render", v)[0]), f"{arm3d}: {name}"); tag(axs[1][2 * k + 1], "SYNTHETIC, unverified", "#b00000")
    show(axs[1][4], crop_white(load_view(f"{P}/{arm3d}/seed0/render", 17)[0]), f"{arm3d}: bottom (unseen)"); tag(axs[1][4], "SYNTHETIC, unverified", "#b00000")
    al = {r["specimen"]: r for r in json.load(open(f"{ROOT}/results/alignment/summary.json"))}[fail]
    show(axs[2][0], img(f"{ROOT}/data/inputs/{fail}/image.png"), f"failure {fail.split('__')[-1]}: photo"); tag(axs[2][0], "REAL", "#006400")
    show(axs[2][1], img(f"{ROOT}/results/alignment/{fail}/A0.png"), f"SAM 3D: ring (IoU {al['A0_iou']:.2f})"); tag(axs[2][1], "SYNTHETIC - FAILED", "#b00000")
    show(axs[2][2], crop_white(load_view(f"{ROOT}/results/phaseA/{fail}/D_render", 16)[0]) if os.path.exists(f"{ROOT}/results/phaseA/{fail}/D_render") else None, "same asset, view 16")
    tag(axs[2][2], "SYNTHETIC - FAILED", "#b00000")
    axs[2][3].axis("off"); axs[2][4].axis("off")
    axs[2][3].text(0.0, 0.5, "What 3D adds here (internal only): one asset gives\nmutually consistent views and a measurable,\n"
                   "KB-traced shape change (profile/side views)\nthat a 2D edit of the photo cannot express.\n\n"
                   "What it does NOT show: that the real leaf\nlooks like this from the side/below; that the\n"
                   "puckering is biologically right (no expert\nreview); any downstream benefit (not run).",
                   fontsize=4.6, va="center", transform=axs[2][3].transAxes)
    save(fig, f"fig10_added_value_{sid}", [f"data/inputs/{sid}", f"results/phaseA/{sid}", f"results/alignment/{fail}"],
         "Added value of a KB-edited 3D asset vs 2D editing. Row 1: all at the input camera - the 3D geometric edit is barely visible "
         "from the photo's viewpoint. Row 2: views absent from the photo, where the edit is visible; these are synthetic and unverified. "
         "Row 3: a reconstruction failure (curled leaf reconstructed as a closed ring).")


if __name__ == "__main__":
    main()
