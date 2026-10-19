"""Render-space metrics of an edited asset vs the unedited asset D on the fixed rig (same gsplat renderer, same cameras).

Geometry (unit-free; renders are not calibrated):
  sil_change      1 - IoU(alpha_X > .5, alpha_D > .5), mean over views
  depth_change    median |z_X - z_D| / rig radius over pixels covered in both
Appearance:
  dE_obj          mean CIEDE76 (Lab L2) between X and D over pixels covered in both
  grain_ratio     Laplacian energy inside the object, X / D   (>1 = grainier)       [QC, brief s20]
  fuzz_ratio      soft-alpha band (0.05 < a < 0.95) area per silhouette perimeter, X / D (>1 = fuzzier edges) [QC]
usage (any env with numpy/scipy/skimage):  view_metrics.py <D_render_dir> <X_render_dir> <rig.json> [out.json]
"""
import os, sys, json, glob
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab


sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rendering.rig_render import load_view as load


def per_view(D, X, radius):
    (rd, ad, zd), (rx, ax, zx) = D, X
    md, mx = ad > 0.5, ax > 0.5; both = md & mx; uni = md | mx
    lap = lambda im, m: float(np.mean(ndimage.laplace(im.mean(-1).astype(np.float32))[ndimage.binary_erosion(m, iterations=2)] ** 2)) if m.sum() > 50 else np.nan
    band = lambda a, m: ((a > 0.05) & (a < 0.95)).sum() / max(1, (m ^ ndimage.binary_erosion(m)).sum())
    za = np.where(ad > 1e-3, zd / np.maximum(ad, 1e-3), 0); zb = np.where(ax > 1e-3, zx / np.maximum(ax, 1e-3), 0)
    return dict(sil_iou=float(both.sum() / max(1, uni.sum())),
                depth_change=float(np.median(np.abs(za - zb)[both]) / radius) if both.sum() > 50 else np.nan,
                dE_obj=float(np.linalg.norm(rgb2lab(rd / 255.0)[both] - rgb2lab(rx / 255.0)[both], axis=1).mean()) if both.sum() > 50 else np.nan,
                grain_ratio=lap(rx, mx) / max(lap(rd, md), 1e-9), fuzz_ratio=band(ax, mx) / max(band(ad, md), 1e-9))


def compare(Ddir, Xdir, rig):
    ids = [v["id"] for v in rig["views"]]; rows = [dict(view=i, **per_view(load(Ddir, i), load(Xdir, i), rig["radius"])) for i in ids]
    agg = {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if k != "view"}
    agg["sil_change"] = 1 - agg.pop("sil_iou")
    return dict(summary=agg, per_view=rows)


if __name__ == "__main__":
    r = compare(sys.argv[1], sys.argv[2], json.load(open(sys.argv[3])))
    if len(sys.argv) > 4: json.dump(r, open(sys.argv[4], "w"), indent=1)
    print(json.dumps({k: round(v, 4) for k, v in r["summary"].items()}))
