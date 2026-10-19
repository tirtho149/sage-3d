"""Review item B4: when does a 3D two-surface leaf beat a two-layer 2D representation?

A two-layer 2D representation (front image = photo, back image = a separately edited
underside) places a lower-surface symptom as correctly as the 3D two-surface leaf in any
view that shows ONE surface. It must be wrong in views that show BOTH surfaces at once:
there, whichever layer it shows, the pixels of the other surface are the wrong surface.

So the measurable 3D advantage for "which surface is the symptom on" is the share of
pixels belonging to the minority surface in each view. Measured for all 12 pilot leaves
(unedited reconstruction D, single sheet, side by the adaxial normal vs each camera) over
the 18 rig views, plus the Tomato two-surface assets (layer ids) as a check.

Also recorded, honestly: in the Test-3 editor the lower mold FOOTPRINT is the upper spot
footprint by construction (KB: mold "below leaf spots"); only its texture differs. Leaf
thickness (2 delta = 0.004 L) is an assumption, not a measurement.

usage (sam3d env): CUDA_VISIBLE_DEVICES=1 LIDRA_SKIP_INIT=true ~/micromamba/envs/sam3d/bin/python experiments/review/b4_surface_visibility.py
  -> results/review/b4_surfaces/summary.{json,md}
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/experiments/demos")
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
from geometry_editor.anatomy import leaf_frame  # noqa: E402
from rendering.rig_render import render_rig  # noqa: E402
from metrics import surfaces  # noqa: E402
from run_demos import rig_of  # noqa: E402

OUT = f"{ROOT}/results/review/b4_surfaces"
LEAVES = [f"{c}__{i}" for c, ids in [("Peach__Leaf_Curl", ["0006", "0063", "0087"]),
                                      ("Tomato__Leaf_Mold", ["0015", "0052", "0092"]),
                                      ("Citrus__Canker", ["0004", "0012", "0016"]),
                                      ("Mango__Anthracnose", ["0044", "0049", "0094"])] for i in ids]
MIXED = 0.10      # a view "shows both surfaces" if the minority surface covers > 10% of leaf pixels


def per_view(side_imgs):
    out = []
    for sd in side_imgs:
        sd = sd.astype(np.int32)                         # uint8 + 40 overflows (255 + 40 -> 39)
        leaf = sd.sum(-1) < 735                          # non-white: leaf pixels only
        up = (leaf & (sd[..., 0] > sd[..., 2] + 40)).sum(); lo = (leaf & (sd[..., 2] > sd[..., 0] + 40)).sum()
        tot = up + lo
        out.append(dict(upper_px=int(up), lower_px=int(lo), minority_frac=float(min(up, lo) / tot) if tot else 0.0))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for sid in LEAVES:
        F = leaf_frame(sid); rig = rig_of(sid); cams = surfaces.camera_centres(rig)
        ids = [v["id"] for v in rig["views"]]
        imgs = [render_rig(F["ply"], rig, colors=surfaces.side_colours_single(F["X"], F["n"], cams[v]), views=[v])[0]["rgb"]
                for v in ids]
        pv = per_view(imgs)
        mf = np.array([p["minority_frac"] for p in pv])
        res[sid] = dict(views=pv, mean_minority_frac=float(mf.mean()), mixed_views=int((mf > MIXED).sum()), n_views=len(pv))
        print(sid, f"minority-surface share {mf.mean():.3f}, mixed views {(mf > MIXED).sum()}/{len(pv)}", flush=True)
    for sid in ("Tomato__Leaf_Mold__0015", "Tomato__Leaf_Mold__0052", "Tomato__Leaf_Mold__0092"):
        lay = f"{ROOT}/results/demos/tomato/{sid}/BLk_s0/asset.ply"
        if os.path.exists(lay) and os.path.exists(lay + ".layers.npy"):
            L = np.load(lay + ".layers.npy"); c = np.where(L[:, None] == 0, [1.0, 0, 0], [0, 0, 1.0])
            imgs = [r["rgb"] for r in render_rig(lay, rig_of(sid), colors=c, views=list(range(18)))]
            mf = np.array([p["minority_frac"] for p in per_view(imgs)])
            res[sid]["two_surface_asset"] = dict(mean_minority_frac=float(mf.mean()), mixed_views=int((mf > MIXED).sum()))
    json.dump(dict(mixed_threshold=MIXED, leaves=res), open(f"{OUT}/summary.json", "w"), indent=1)
    L = ["# B4: share of pixels a two-layer 2D representation must get wrong", "",
         f"Per rig view, the minority surface's share of leaf pixels (upper vs lower, by the adaxial normal). A two-layer",
         f"2D representation shows one layer per view, so this share is unavoidably the wrong surface. Mixed view = share > {MIXED}.", "",
         "| leaf | mean minority share | mixed views / 18 | two-surface asset (mixed views) |", "|---|---|---|---|"]
    for sid, r in res.items():
        tw = r.get("two_surface_asset")
        L.append(f"| {sid} | {r['mean_minority_frac']:.3f} | {r['mixed_views']}/{r['n_views']} | "
                 f"{(str(tw['mixed_views']) + '/18') if tw else '-'} |")
    L += ["", "Mold footprint on the lower surface = upper spot footprint by construction (KB 'below leaf spots'); only texture differs.",
          "Leaf thickness 2*delta = 0.004 L is an assumption (bilayer.py), not a measurement."]
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
