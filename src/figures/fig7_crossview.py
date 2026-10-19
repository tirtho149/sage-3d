"""Figure 7 - Cross-view consistency: independent per-view 2D edits (Bmv generic, Cmv KB) of the 8 ring renders vs one
edited 3D asset seen from the same 8 views. Metric: correspondence-based (src/metrics/crossview.py) - median CIELAB dE of
the SAME surface point between adjacent views (lower = more consistent; unedited D = resampling floor).
An earlier mean-colour-shift consistency score was dropped: it rated per-view 2D edits as consistent as 3D edits because it
ignores point correspondence (uninformative). INTERNAL consistency only: nothing about agreement with the real specimen."""

import os, sys, json
import numpy as np, matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures.common import ROOT, TW, show, save, SYN
from rendering.rig_render import load_view
from metrics.appearance_kb import lab_delta
from metrics.crossview import main as corr_main


def consistency(V):
    V = np.array(V); vb = V.mean(0); return float(1 - np.linalg.norm(V - vb, axis=1).mean() / max(np.linalg.norm(vb), 1e-9))


def per_view_2d(sid, arm):
    D = f"{ROOT}/results/pilot/{sid}/D/images"; X = f"{ROOT}/results/pilot/{sid}/{arm}_masked"; out, ims = [], []   # object-masked (audit F-2D-2)
    for i in range(8):
        if not os.path.exists(f"{X}/{i:03d}.png"): return None, None
        a = np.asarray(Image.open(f"{D}/{i:03d}.png").convert("RGB"), float); b = np.asarray(Image.open(f"{X}/{i:03d}.png").convert("RGB").resize(a.shape[1::-1]), float)
        m = (a < 250).any(-1) & (b < 250).any(-1); s = lab_delta(a, b, m); out.append([s["dL"], s["da"], s["db"]]); ims.append(b.astype(np.uint8))
    return out, ims


def per_view_3d(sid, arm):
    P = f"{ROOT}/results/phaseA/{sid}"; X = f"{P}/{arm}/seed0/render"; out, ims = [], []
    if not os.path.exists(f"{X}/rgb/007.png"): return None, None
    for i in range(8):
        a, aa, _ = load_view(f"{P}/D_render", i); b, ab, _ = load_view(X, i); s = lab_delta(a.astype(float), b.astype(float), (aa > .5) & (ab > .5))
        out.append([s["dL"], s["da"], s["db"]]); ims.append(b)
    return out, ims


def main(sid="Peach__Leaf_Curl__0087"):
    CORR = corr_main(sid); rows = []
    for arm, kind in (("Bmv", "2D per-view (generic)"), ("Cmv", "2D per-view (KB)"), ("E", "3D GE generic"), ("F", "3D GE KB"), ("K", "3D GE KB, geo frozen"), ("H", "3D geometry + KB appearance")):
        V, ims = (per_view_2d if arm.endswith("mv") else per_view_3d)(sid, arm)
        if V is not None: rows.append((arm, kind, CORR.get(arm), ims))
    fig, axs = plt.subplots(len(rows), 9, figsize=(TW, TW * len(rows) / 9 * 1.05), squeeze=False, gridspec_kw=dict(width_ratios=[1] * 8 + [1.4]))
    for r, (arm, kind, c, ims) in enumerate(rows):
        for i in range(8):
            a = ims[i]; m = (a < 250).any(-1); ys, xs = np.nonzero(m)
            show(axs[r][i], a[max(0, ys.min() - 6):ys.max() + 6, max(0, xs.min() - 6):xs.max() + 6] if len(ys) else a, f"view {i}" if r == 0 else None)
        axs[r][8].axis("off"); axs[r][8].text(0.02, 0.5, f"{arm}: {kind}\nsame-point dE {c:.1f}\n(floor D {CORR['D (floor)']:.1f})", fontsize=4.8, va="center", transform=axs[r][8].transAxes)
        axs[r][8].text(0.02, 0.1, "SYNTHETIC", transform=axs[r][8].transAxes, **SYN)
    save(fig, f"fig7_crossview_{sid}", [f"results/pilot/{sid}/Bmv", f"results/pilot/{sid}/Cmv", f"results/phaseA/{sid}/*/seed0/render"],
         "Cross-view consistency of the edit: per-view 2D edits are made independently on each render; a 3D edit is one asset. "
         "Consistency is internal agreement between views only (no real multi-view reference exists).")
    return {a: c for a, _, c, _ in rows}


if __name__ == "__main__":
    print(main(*sys.argv[1:2]))
