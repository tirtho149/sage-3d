"""Figure 2 - Reconstruction debugging: input, SAM mask, SAM 3D at input camera before (A0) and after (A3*) pose correction."""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures.common import ROOT, TW, img, show, save


def overlay(inp, mask, render):
    a = img(inp).astype(float); m = np.asarray(Image.open(mask).convert("L")) > 127
    r = img(render); pr = (r < 250).any(-1) if r is not None else np.zeros_like(m)
    o = a * 0.45 + 140; o[m & ~pr] = [40, 90, 255]; o[pr & ~m] = [255, 60, 40]; o[pr & m] = [60, 170, 60]
    return o.clip(0, 255).astype(np.uint8)


def main():
    S = json.load(open(f"{ROOT}/results/alignment/summary.json")); n = len(S); half = (n + 1) // 2
    fig, axs = plt.subplots(half, 8, figsize=(TW, TW * half / 8 * 0.95))
    for k, r in enumerate(S):
        s = r["specimen"]; row, c0 = k % half, 4 * (k // half); A = f"{ROOT}/results/alignment/{s}"; I = f"{ROOT}/data/inputs/{s}"
        star = "A3_render_compare" if r["selected"] == "A3_render_compare" else "A2_closed_form"
        crop, dis, num = s.split("__"); short = f"{crop} {dis.split('_')[0].lower()} {num}"
        show(axs[row][c0], img(f"{I}/image.png"), short)
        show(axs[row][c0 + 1], img(f"{A}/A0.png"), "A0 render")
        show(axs[row][c0 + 2], overlay(f"{I}/image.png", f"{I}/0.png", f"{A}/A0.png"), f"A0 IoU {r['A0_iou']:.2f}")
        show(axs[row][c0 + 3], overlay(f"{I}/image.png", f"{I}/0.png", f"{A}/{star}.png"),
             f"A3* IoU {r['A3s_iou']:.2f}" + (" (A2)" if star.startswith("A2") else ""), note="RECON. FAILURE" if r["label"] != "ok" else None)
    for k in range(n, half * 2):
        for c in range(4): axs[k % half][4 * (k // half) + c].axis("off")
    for ax in axs.flat: ax.title.set_fontsize(4.3)
    save(fig, "fig2_alignment", ["results/alignment/summary.json", "results/alignment/*/A0.png", "results/alignment/*/A3_render_compare.png"],
         "Input-view alignment of SAM 3D reconstructions. Overlays: green = render and mask agree, blue = mask only, red = render only. "
         "A0 = SAM 3D pose; A3* = pose-only render-and-compare (geometry frozen) accepted only if better than the closed-form A2. "
         "Fitted to the input silhouette: agreement here is NOT reconstruction accuracy (no second real view exists).")


if __name__ == "__main__":
    main()
