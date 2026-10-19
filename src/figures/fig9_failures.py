"""Figure 9 - Failure taxonomy with real artifacts (technical failures first, then baseline/biological-plausibility failures)."""
import os, sys, json
import numpy as np, matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures.common import ROOT, TW, img, show, save, crop_white
from rendering.rig_render import load_view


def zoom(a, frac=0.35, cy=0.5, cx=0.5):
    H, W = a.shape[:2]; h, w = int(H * frac), int(W * frac); y, x = int(cy * H - h / 2), int(cx * W - w / 2)
    return a[max(0, y):y + h, max(0, x):x + w]


def main():
    al = {r["specimen"]: r for r in json.load(open(f"{ROOT}/results/alignment/summary.json"))}
    P = f"{ROOT}/results/pilot"; A = f"{ROOT}/results/alignment"; Q = f"{ROOT}/results/phaseA"
    panels = [
        (img(f"{ROOT}/data/inputs/Peach__Leaf_Curl__0006/image.png"), "1a Peach 0006 photo (curled leaf)"),
        (crop_white(img(f"{A}/Peach__Leaf_Curl__0006/A0.png")), f"1b SAM 3D: closed ring\nIoU {al['Peach__Leaf_Curl__0006']['A0_iou']:.2f}; best fix {al['Peach__Leaf_Curl__0006']['A3s_iou']:.2f}"),
        (img(f"{A}/Peach__Leaf_Curl__0063/overlay_A3_render_compare.png"), f"2 pose refinement A3 diverged\n(0063: IoU {al['Peach__Leaf_Curl__0063']['A3_iou']:.2f}; A2 kept)"),
        (zoom(load_view(f"{Q}/Peach__Leaf_Curl__0087/D_render", 0)[0]), "3a D (unedited) close-up"),
        (zoom(load_view(f"{Q}/Peach__Leaf_Curl__0087/E/seed0/render", 0)[0]), "3b GE arm E close-up:\ngrain x13.6, fuzz x1.9, jitter"),
        (crop_white(img(f"{P}/Peach__Leaf_Curl__0006/D/images/000.png")), "4a 0006 D render"),
        (crop_white(img(f"{P}/Peach__Leaf_Curl__0006/E_INVALID_nan_opacity_noop/seed0/views/images/000.png")), "4b 'edited' (pre-fix): bit-identical\n(inf opacity -> NaN -> fp16 steps skipped)"),
        (img(f"{P}/Peach__Leaf_Curl__0087/Bmv/000.png"), "5 per-view 2D edit run unmasked:\nIP2P invents a new scene"),
        (img(f"{ROOT}/data/inputs/Peach__Leaf_Curl__0087/image.png"), "6a 0087 photo: puckering visible"),
        (img(f"{Q}/Peach__Leaf_Curl__0087/I/seed0.png"), "6b arm I (2D shape-capable, KB):\nflat leaf replaces real puckering"),
    ]
    fig, axs = plt.subplots(2, 5, figsize=(TW, TW * 0.5))
    for ax, (a, t) in zip(axs.flat, panels): show(ax, a, t)
    for ax in axs.flat: ax.title.set_fontsize(4.4)
    save(fig, "fig9_failures", ["results/alignment", "results/pilot", "results/phaseA", "results/ge_diagnosis"],
         "Failure taxonomy (all from real runs). 1 reconstruction failure; 2 pose-refinement divergence (handled by the A3* acceptance rule); "
         "3 GaussianEditor appearance edit adds grain/fuzz and incoherent jitter, no deformation; 4 GE no-op caused by non-finite opacity (fixed); "
         "5 unmasked per-view 2D baseline (fixed by masking); 6 a 2D edit that may change shape replaces observed puckering with a flat leaf.")


if __name__ == "__main__":
    main()
