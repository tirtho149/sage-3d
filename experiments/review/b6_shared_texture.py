"""Review item B6: is 3D consistency a property of OUR editor or of sharing one asset?

Baseline "Ctex" (shared canonical texture): take the 2D KB edit of the input photo (arm C,
seed 0, InstructPix2Pix), lift its colours onto the unedited 3D asset -- every Gaussian
visible at the corrected input camera takes the edited pixel colour, hidden ones keep their
colour -- and render the rig. No geometry change, no knowledge beyond the 2D edit. It is
multi-view consistent by construction, like any single shared asset.

Reported per specimen with the same metrics as E4: cross-view same-point dE (crossview.score,
8 ring views) and edit size vs the unedited render D (mean Lab dE over object pixels, and
silhouette change). If Ctex is as consistent as the geometry editor while recolouring more,
consistency alone cannot credit the KB geometry algorithm.

usage (sam3d env): CUDA_VISIBLE_DEVICES=1 LIDRA_SKIP_INIT=true ~/micromamba/envs/sam3d/bin/python experiments/review/b6_shared_texture.py
  -> results/review/b6_shared_texture/<sid>/asset.ply, summary.{json,md}
"""
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from skimage.color import rgb2lab

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src")
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
from geometry_editor.anatomy import input_camera, rgb_to_dc  # noqa: E402
from rendering.rig_render import render_rig  # noqa: E402
from metrics.crossview import pts, score  # noqa: E402

OUT = f"{ROOT}/results/review/b6_shared_texture"


def lift(sid):
    ply = f"{ROOT}/results/pilot/sam3d/{sid}/seed0/splat_ge.ply"
    edit = f"{ROOT}/results/pilot/{sid}/C/seed0.png"
    if not os.path.exists(edit):
        return None, "no 2D KB edit (C) at the input view"
    cam = input_camera(sid, ply)
    img = np.asarray(Image.open(edit).convert("RGB").resize((cam["W"], cam["H"])), np.float64) / 255.0
    uvw = cam["Xc"] @ np.asarray(cam["K"]).T
    px = np.clip(np.round(uvw[:, 0] / uvw[:, 2]).astype(int), 0, cam["W"] - 1)
    py = np.clip(np.round(uvw[:, 1] / uvw[:, 2]).astype(int), 0, cam["H"] - 1)
    vis = cam["visible"]
    pd = PlyData.read(ply); v = pd["vertex"].data.copy()
    dc = rgb_to_dc(img[py[vis], px[vis]])
    for k in range(3):
        v[f"f_dc_{k}"][vis] = dc[:, k].astype(v[f"f_dc_{k}"].dtype)
    d = f"{OUT}/{sid}"; os.makedirs(d, exist_ok=True)
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(f"{d}/asset.ply")
    return f"{d}/asset.ply", f"lifted {int(vis.sum())} visible Gaussians of {len(vis)}"


def renders(ply, rig):
    R = render_rig(ply, rig, views=list(range(8)))
    return ([r["rgb"].astype(float) for r in R], [r["depth"] for r in R], [r["alpha"] for r in R], R)


def edit_size(Rd, Rx):
    de, sil = [], []
    for a, b in zip(Rd, Rx):
        ma, mb = a["alpha"] > 0.5, b["alpha"] > 0.5; both = ma & mb
        sil.append(1 - both.sum() / max((ma | mb).sum(), 1))
        if both.sum() > 50:
            de.append(np.linalg.norm(rgb2lab(a["rgb"][both] / 255.0) - rgb2lab(b["rgb"][both] / 255.0), axis=1).mean())
    return float(np.mean(de)) if de else None, float(np.mean(sil))


def main():
    os.makedirs(OUT, exist_ok=True)
    sids = sorted(os.path.basename(p) for p in glob.glob(f"{ROOT}/results/pilot/*__*") if os.path.isdir(p) and "sam3d" not in p)
    res = {}
    for sid in sids:
        rigp = f"{ROOT}/results/pilot/{sid}/D/rig.json"
        if not os.path.exists(rigp):
            continue
        rig = json.load(open(rigp))
        ply, note = lift(sid)
        if ply is None:
            res[sid] = dict(status=note); continue
        D0 = f"{ROOT}/results/pilot/sam3d/{sid}/seed0/splat_ge.ply"
        imgsD, dD, aD, RD = renders(D0, rig); imgsX, dX, aX, RX = renders(ply, rig)
        floor = score(pts(D0), imgsD, dD, aD, rig)["median_dE_adjacent"]
        ctex = score(pts(ply), imgsX, dX, aX, rig)["median_dE_adjacent"]
        dE, sil = edit_size(RD, RX)
        prev = {}
        cj = f"{ROOT}/results/statistics/crossview_corr_{sid}.json"
        if os.path.exists(cj):
            prev = {k: v.get("median_dE_adjacent") for k, v in json.load(open(cj)).items()}
        res[sid] = dict(note=note, crossview_floor_D=floor, crossview_Ctex=ctex, edit_dE_obj=dE, edit_sil_change=sil,
                        crossview_previous=prev)
        print(sid, f"floor {floor:.1f}  Ctex {ctex:.1f}  edit dE {dE:.1f}  | prev {prev}", flush=True)
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    ok = {k: v for k, v in res.items() if "crossview_Ctex" in v}
    med = lambda key: float(np.median([v[key] for v in ok.values() if v.get(key) is not None]))
    medp = lambda arm: float(np.median([v["crossview_previous"][arm] for v in ok.values()
                                        if v["crossview_previous"].get(arm) is not None])) if any(
        v["crossview_previous"].get(arm) is not None for v in ok.values()) else None
    L = ["# B6: shared-canonical-texture baseline (2D KB edit lifted onto one 3D asset)", "",
         f"n = {len(ok)} specimens. Cross-view = median same-point dE over adjacent ring views (lower = more consistent).", "",
         "| method | cross-view dE (median over specimens) | edit size: colour dE vs D | silhouette change |", "|---|---|---|---|",
         f"| D unedited (floor) | {med('crossview_floor_D'):.1f} | 0 | 0 |",
         f"| **Ctex: 2D KB edit lifted onto the shared asset** | **{med('crossview_Ctex'):.1f}** | {med('edit_dE_obj'):.1f} | {med('edit_sil_change'):.2f} |"]
    for arm, lab in (("Cmv", "per-view 2D KB edits"), ("E", "GaussianEditor generic"), ("F", "GaussianEditor + KB"), ("J", "geometry editor (KB)")):
        v = medp(arm)
        if v is not None:
            L.append(f"| {lab} ({arm}, E4) | {v:.1f} | see slide E4 | see slide E4 |")
    L += ["", "Ctex needs no 3D editing and no geometry knowledge; its consistency comes from sharing one asset.",
          "Hidden (back-facing) Gaussians keep their colour: the lifted edit covers only what the photo shows."]
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
