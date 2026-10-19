"""Stage 1 generator: one image store for every 2D / 3D arm, view set and rendering condition (sam3d env, one GPU).

Images are deterministic in (arm, asset, view, condition, edit seed) and stored once under
data/synthetic/stage1/images/<arm>/<asset_id>/<view>_<cond>_e<seed>.jpg; experiments select rows from
data/synthetic/stage1/manifest_<arm>.csv. Rendering, pose, plates, crops and photometrics are Stage-0's
(render3d.py / build2d.py), so Stage-0R differs from Stage 0 only where intended.

Arms (texture source x geometry), all from the same parent asset:
  2D  PHOTO2D  photo                     | GEN2D  generic IP2P edit     | KB2D  KB IP2P edit
      PLATE2D  background-only control: the photo with the organ inpainted, same crops/variants (shortcut test)
  3D  U3D      SAM 3D colours, no ops     | TEX3D  photo lifted           | GEN3D generic edit lifted
      KBAPP3D  KB edit lifted, no ops     | GEO3D  KB geometry ops only   | KB3D  KB ops + KB edit lifted
      TEXGEO3D KB ops + photo lifted (= what Stage-0 "KB3D" actually was, because of the edit bug)
      HARM3D   SAM 3D colours globally colour-transferred to the parent photo (Lab mean/std inside the organ, fitted
               at the input view, applied to every view): fixes the colour shift without copying any lesion pixel
Lifting = colour of Gaussians visible at the input camera <- edit pixel (render3d.lift_colors); hidden ones unchanged.
Edits come from data/synthetic/stage1/edits2d (fixed uint8-mask IP2P, seeds 0-2).
View sets (yaw, pitch in degrees about the object centroid; (0,0) = corrected input camera):
  V8      Stage-0 set: (0,0) (+-15,0) (0,+-15) (+-30,0) (0,25)
  NARROW  (0,0) and 7 perturbations within 8 degrees
  WIDE    (0,0) (+-45,0) (0,+-40) (+-90,0) (180,0)   (novel / extreme views, incl. the back)
2D "views" are Stage-0's seeded 2D variants (rotation/scale/shift/flip), indexed v0..v7.
Replicates: --edit-seeds r1,r2 renders replicate r of every arm (edit arms use edit seed r; all arms use nuisance
draws seeded by r), so replicate r is matched across arms.
Nuisance randomness (2D variant, background plate, photometrics) is seeded by (2D|3D, asset, view, cond, edit seed),
NOT by arm: arms from one parent differ only in texture/geometry. Verified: U3D and TEXGEO3D R0/R1 renders equal
Stage-0 U3D / "KB3D" pixel for pixel (mean |diff| 0.0).
usage: CUDA_VISIBLE_DEVICES=1 LIDRA_SKIP_INIT=true ~/micromamba/envs/sam3d/bin/python -m src.generation.s1_arms \
         --arms U3D,TEX3D --views V8 --conds R0,R1,R2,R5 --edit-seeds 0 [--assets a,b] [--stage stage0]
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render3d as R  # noqa: E402
from build2d import variant  # noqa: E402

S1 = f"{ROOT}/data/synthetic/stage1"
from s1_const import VIEWSETS, ARMS3D, ARMS2D, CROSS, vkey  # noqa: E402,F401
FIELDS = ["path", "arm", "class_id", "asset_id", "root_sha1", "view", "yaw", "pitch", "cond", "edit_seed", "bg_asset",
          "input_iou", "geo_ops", "lift_frac"]


def source(kind, a, img0, seed, plates=None):
    if kind == "photo":
        return img0
    if kind == "kbx":
        p = f"{S1}/edits2d/{a['asset_id']}/kbx_s{seed}.png"
        return np.asarray(Image.open(p).convert("RGB").resize((img0.shape[1], img0.shape[0]))) if os.path.exists(p) else None
    if kind == "plate":                          # background only: the parent photo with the organ inpainted (Telea)
        return plates[a["asset_id"]]
    p = f"{S1}/edits2d/{a['asset_id']}/{kind}_s{seed}.png"
    if not os.path.exists(p):
        return None
    return np.asarray(Image.open(p).convert("RGB").resize((img0.shape[1], img0.shape[0])))


def lab_transfer_params(render_rgb, render_alpha, photo, mask):
    from skimage.color import rgb2lab
    A = rgb2lab(np.clip(render_rgb, 0, 1))[render_alpha > 0.5]; B = rgb2lab(photo / 255.0)[mask]
    return A.mean(0), A.std(0) + 1e-6, B.mean(0), B.std(0)


def lab_apply(rgb, prm):
    from skimage.color import lab2rgb, rgb2lab
    ma, sa, mb, sb = prm
    return np.clip(lab2rgb((rgb2lab(np.clip(rgb, 0, 1)) - ma) / sa * sb + mb), 0, 1)


def compose_cond(im, alpha, cond, a, plates, ids, cls_of, rng):
    """Stage-0 conditions: R0 white, R1 own plate, R2 other-class plate, R5 = R2 + photometric. alpha HxW in [0,1]."""
    al = alpha[..., None]
    if cond == "R0":
        return (im * al + 255 * (1 - al)).astype(np.uint8), ""
    if cond == "R1":
        bg_id = a["asset_id"]
    else:
        others = [i for i in ids if cls_of[i] != a["class_id"]]
        bg_id = others[int(rng.integers(len(others)))]
    bg = np.asarray(Image.fromarray(plates[bg_id]).resize((im.shape[1], im.shape[0]))).astype(np.float32)
    out = (im.astype(np.float32) * al + bg * (1 - al)).clip(0, 255).astype(np.uint8)
    if cond == "R5":
        out = R.photometric(out, rng)
    return out, bg_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True); ap.add_argument("--views", default="V8"); ap.add_argument("--conds", default="R0,R1,R2,R5")
    ap.add_argument("--edit-seeds", default="0"); ap.add_argument("--assets", default=""); ap.add_argument("--stage", default="stage0")
    args = ap.parse_args()
    from src.generation.registry import assets
    A_all = assets(args.stage); A = A_all[A_all.splat.notna()] if any(x in ARMS3D for x in args.arms.split(",")) else A_all
    if args.assets:
        A = A[A.asset_id.isin(args.assets.split(","))]
    import json
    K = json.load(open(f"{ROOT}/data/synthetic/stage0/kb_specs.json"))
    if os.path.exists(f"{S1}/kb_specs_all.json"):
        K = {**json.load(open(f"{S1}/kb_specs_all.json")), **K}          # Stage-0 specs win for Stage-0 classes
    plates = {}
    for _, a in A_all.iterrows():
        img, m = R.load_inputs(a); plates[a.asset_id] = R.plate(img, m)
    ids = sorted(plates); cls_of = dict(zip(A_all.asset_id, A_all.class_id))
    views = VIEWSETS[args.views]; conds = args.conds.split(","); seeds = [int(s) for s in args.edit_seeds.split(",")]
    for arm in args.arms.split(","):
        man = f"{S1}/manifest_{arm}.csv"; done = set()
        if os.path.exists(man):
            with open(man) as f:
                done = {(r["asset_id"], r["view"], r["cond"], r["edit_seed"]) for r in csv.DictReader(f)}
        with open(man, "a", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=FIELDS)
            if not done:
                w.writeheader()
            for _, a in A.iterrows():
                a = a.to_dict(); img0, m = R.load_inputs(a); H, W = m.shape
                label = a["class_id"]
                if arm in CROSS:
                    from edit2d_sibling import SIBLING
                    label = SIBLING.get(a["class_id"])
                    if label is None or K[label]["target_organ"] != K[a["class_id"]]["target_organ"]:
                        continue
                for sd in seeds:          # replicate r: edit seed r for edit arms; for the others only the nuisance draws change
                    todo = [(y, p, c) for (y, p) in views for c in conds
                            if (a["asset_id"], vkey(y, p) if arm in ARMS3D else f"v{views.index((y, p))}", c, str(sd)) not in done]
                    if not todo:
                        continue
                    if arm in ARMS2D:
                        src = source(ARMS2D[arm], a, img0, sd, plates)
                        if src is None:
                            print("skip (no edit)", arm, a["asset_id"], sd, flush=True); continue
                        for vi in range(len(views)):
                            rng = np.random.default_rng(R.seed_of("2D", a["asset_id"], vi, "geo", sd))
                            im, mk = variant(src, m, rng) if vi else (src, m)
                            x0, y0, x1, y1 = R.crop_box(mk.astype(np.float32))
                            for c in conds:
                                r2 = np.random.default_rng(R.seed_of("2D", a["asset_id"], vi, c, sd))
                                out, bg = compose_cond(im, mk.astype(np.float32), c, a, plates, ids, cls_of, r2)
                                p = f"{S1}/images/{arm}/{a['asset_id']}/v{vi}_{c}_e{sd}.jpg"; R.save(out[y0:y1, x0:x1], p)
                                w.writerow(dict(path=os.path.relpath(p, ROOT), arm=arm, class_id=label, asset_id=a["asset_id"],
                                                root_sha1=a["root_sha1"], view=f"v{vi}", yaw="", pitch="", cond=c, edit_seed=sd, bg_asset=bg,
                                                input_iou="", geo_ops="", lift_frac=""))
                    else:
                        tex, geo = ARMS3D[arm]
                        Al = R.alignment(a, os.path.dirname(a["image"])); Kt = torch.tensor(Al["K"], dtype=torch.float32, device="cuda")
                        ply, ops = a["splat"], []
                        if geo:
                            ga = dict(a, asset_id=a["asset_id"] + ("__x" if arm in CROSS else ""))     # separate cache for cross ops
                            gp, ops = R.geometry_edit(ga, K[label]) if label in K else (None, [])
                            ply = gp or a["splat"]
                        G, sel = R.posed(a, ply, Al); lf = ""
                        harm = None
                        if tex == "harm":
                            r0, al0, _ = R.render(G, Kt, W, H); harm = lab_transfer_params(r0, al0, img0, m); tex = None
                        if tex:
                            src = source(tex, a, img0, sd)
                            if src is None:
                                print("skip (no edit)", arm, a["asset_id"], sd, flush=True); continue
                            G["rgb"], lf = R.lift_colors(G, Kt, W, H, src)
                        _, a0, _ = R.render(G, Kt, W, H); iou = float(((a0 > 0.5) & m).sum() / max(((a0 > 0.5) | m).sum(), 1))
                        for (yaw, pitch) in views:
                            rgb, alpha, _ = R.render(R.rotate(G, yaw, pitch) if (yaw or pitch) else G, Kt, W, H)
                            if harm is not None:
                                rgb = lab_apply(rgb, harm)
                            x0, y0, x1, y1 = R.crop_box(alpha)
                            for c in conds:
                                rng = np.random.default_rng(R.seed_of("3D", a["asset_id"], vkey(yaw, pitch), c, sd))
                                out, bg = compose_cond((rgb * 255), alpha, c, a, plates, ids, cls_of, rng)
                                p = f"{S1}/images/{arm}/{a['asset_id']}/{vkey(yaw, pitch)}_{c}_e{sd}.jpg"; R.save(out[y0:y1, x0:x1], p)
                                w.writerow(dict(path=os.path.relpath(p, ROOT), arm=arm, class_id=label, asset_id=a["asset_id"],
                                                root_sha1=a["root_sha1"], view=vkey(yaw, pitch), yaw=yaw, pitch=pitch, cond=c, edit_seed=sd,
                                                bg_asset=bg, input_iou=round(iou, 4), geo_ops="+".join(ops), lift_frac=lf))
                    fo.flush()
                print(arm, a["asset_id"], flush=True)


if __name__ == "__main__":
    main()
