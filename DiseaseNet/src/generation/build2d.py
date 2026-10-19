"""Stage 0 step 6: 2D arms with the same image budget and conditions as the 3D arms (CPU; sam3d env for cv2).

Arms (one source image per parent asset):
  PHOTO   the real parent photo itself (few-shot real reference, same parents as every synthetic arm)
  KB2D    the KB InstructPix2Pix edit of that photo (edits2d/<asset>/kb_s0.png)
  GEN2D   the generic edit ("add disease symptoms to the <organ>")
"Views" = 8 seeded 2D geometric variants (rotation +-20 deg, scale 0.85-1.15, shift, h-flip) instead of
3D viewpoints; conditions R0/R1/R2/R5 as in render3d.py (organ on white / own image / organ on another
class's background plate / R2 + photometric); same object-centred square crop and 384 px output.
usage: ~/micromamba/envs/sam3d/bin/python src/generation/build2d.py -> images/<arm>/..., manifest_2d.csv
"""
import csv
import os
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render3d import plate, photometric, crop_box, save, seed_of, CONDS, ST  # noqa: E402


def variant(img, mask, rng):
    H, W = mask.shape
    ang = rng.uniform(-20, 20); sc = rng.uniform(0.85, 1.15); tx, ty = rng.uniform(-0.06, 0.06, 2) * (W, H)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, sc); M[:, 2] += (tx, ty)
    im = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mk = cv2.warpAffine(mask.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
    if rng.random() < 0.5:
        im, mk = im[:, ::-1].copy(), mk[:, ::-1].copy()
    return im, mk


def main():
    from src.generation.registry import assets
    A = assets()
    plates, base = {}, {}
    for _, a in A.iterrows():
        img = np.asarray(Image.open(a.image).convert("RGB")); m = np.asarray(Image.open(a["mask"]).convert("L")) > 127
        plates[a.asset_id] = plate(img, m); base[a.asset_id] = (img, m)
    ids = sorted(plates); cls_of = dict(zip(A.asset_id, A.class_id))
    path = f"{ST}/manifest_2d.csv"; new = not os.path.exists(path)
    done = set()
    if not new:
        with open(path) as f:
            done = {(r["arm"], r["asset_id"]) for r in csv.DictReader(f)}
    with open(path, "a", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["path", "arm", "class_id", "asset_id", "root_sha1", "view", "cond", "bg_asset", "input_iou", "geo_ops", "lift_frac"])
        if new:
            w.writeheader()
        for _, a in A.iterrows():
            img0, m = base[a.asset_id]
            src = {"PHOTO": img0, "KB2D": f"{ST}/edits2d/{a.asset_id}/kb_s0.png", "GEN2D": f"{ST}/edits2d/{a.asset_id}/generic_s0.png"}
            for arm, s in src.items():
                if (arm, a.asset_id) in done:
                    continue
                if isinstance(s, str):
                    if not os.path.exists(s):
                        continue
                    s = np.asarray(Image.open(s).convert("RGB").resize((img0.shape[1], img0.shape[0])))
                for vi in range(8):
                    rng = np.random.default_rng(seed_of(arm, a.asset_id, vi, "geo"))
                    im, mk = variant(s, m, rng) if vi else (s, m)
                    x0, y0, x1, y1 = crop_box(mk.astype(np.float32))
                    alpha = mk.astype(np.float32)[..., None]
                    for cond in CONDS:
                        r2 = np.random.default_rng(seed_of(arm, a.asset_id, vi, cond))
                        bg_id = ""
                        if cond == "R0":
                            out = (im * alpha + 255 * (1 - alpha)).astype(np.uint8)
                        elif cond == "R1":
                            out = im; bg_id = a.asset_id
                        else:
                            others = [i for i in ids if cls_of[i] != a.class_id]      # class-independent background
                            bg_id = others[int(r2.integers(len(others)))]
                            bg = np.asarray(Image.fromarray(plates[bg_id]).resize((im.shape[1], im.shape[0])))
                            out = (im * alpha + bg * (1 - alpha)).astype(np.uint8)
                            if cond == "R5":
                                out = photometric(out, r2)
                        p = f"{ST}/images/{arm}/{a.asset_id}/v{vi}_{cond}.jpg"; save(out[y0:y1, x0:x1], p)
                        w.writerow(dict(path=os.path.relpath(p, ROOT), arm=arm, class_id=a.class_id, asset_id=a.asset_id,
                                        root_sha1=a.root_sha1, view=vi, cond=cond, bg_asset=bg_id, input_iou="", geo_ops="", lift_frac=""))
                fo.flush()
            print(a.asset_id, flush=True)


if __name__ == "__main__":
    main()
