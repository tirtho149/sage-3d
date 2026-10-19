"""Stage 0 step 4: 2D edits of every parent photo with SAGE-Extend's InstructPix2Pix wrapper
(timbrooks/instruct-pix2pix, guidance 7.5 / image guidance 1.5, 20 steps, composited inside the organ mask).

  kb       : the class's KB joint-stage prompt (verbatim phrases, data/synthetic/stage0/kb_specs.json)
  generic  : "add disease symptoms to the <organ>" (generic control, same seed)
Outputs data/synthetic/stage0/edits2d/<asset_id>/{kb,generic}_s0.png + .json. Resumable.
The KB edit is used twice: as the KB-2D arm, and lifted onto the 3D asset for the KB-3D appearance edit.

usage (Research .venv, GPU 0): CUDA_VISIBLE_DEVICES=0 ~/Desktop/Research/.venv/bin/python -m src.generation.edit2d [--out DIR] [--seeds 0,1]

BUG FIXED 2026-10-16 (Stage 1 audit): Stage 0 passed a *boolean* mask; ip2p_core.edit computes `mask > 127`, which is
all-False for booleans, so every Stage-0 edit returned the unedited photo (all 60 assets; kb == generic byte-identical).
The mask is now passed as uint8 0/255 and every output is checked to differ from the input inside the mask.
Stage-0 outputs (data/synthetic/stage0/edits2d) are kept unchanged as the audited record; fixed edits go to --out.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.expanduser("~/Desktop/Research/external/SAGE-Extend")
sys.path.insert(0, ROOT); sys.path.insert(0, f"{EXT}/src/editing2d")


# cross-class (knowledge-only) arm: each Stage-0 crop has exactly two classes; kbx = the other class's KB prompt
SIBLING = {}
for _c in json.load(open(f"{ROOT}/configs/datasets/stage0_classes.json"))["classes"]:
    SIBLING.setdefault(_c["class_id"].split("::")[0], []).append(_c["class_id"])
SIBLING = {a: b for pair in SIBLING.values() if len(pair) == 2 for a, b in (pair, pair[::-1])}
MIN_DE = 1.0   # mean |RGB difference| inside the mask below this = "no edit" -> error


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/synthetic/stage0/edits2d")
    ap.add_argument("--seeds", default="0"); ap.add_argument("--assets", default="")
    ap.add_argument("--stage", default="stage0", help="registry stage: stage0 | s1all")
    ap.add_argument("--cross", action="store_true", help="write kbx_s<seed>: the SIBLING class's KB prompt (same crop, same organ)")
    args = ap.parse_args(); seeds = [int(x) for x in args.seeds.split(",")]
    from src.generation.registry import assets
    from ip2p_core import edit
    K = json.load(open(f"{ROOT}/data/synthetic/stage0/kb_specs.json"))
    A = assets(args.stage)
    if args.assets:
        A = A[A.asset_id.isin(args.assets.split(","))]
    for _, a in A.iterrows():
        spec = K[a.class_id]; organ = spec["target_organ"]
        prompts = {"kb": spec["prompts"]["kb_joint_stage"]["text"], "generic": f"add disease symptoms to the {organ}"}
        if args.cross:
            sib = SIBLING.get(a.class_id)
            if sib is None or K[sib]["target_organ"] != organ:
                continue
            prompts = {"kbx": K[sib]["prompts"]["kb_joint_stage"]["text"]}
        d = f"{ROOT}/{args.out}/{a.asset_id}"; os.makedirs(d, exist_ok=True)
        img = Image.open(a.image).convert("RGB")
        mask = (np.asarray(Image.open(a["mask"]).convert("L")) > 127).astype(np.uint8) * 255    # uint8, as ip2p_core expects
        for name, pr in prompts.items():
            for sd in seeds:
                out = f"{d}/{name}_s{sd}.png"
                if os.path.exists(out) or not pr:
                    continue
                t = time.time(); res = edit(img, mask, pr, sd)
                im = res[0] if isinstance(res, tuple) else res
                im = im if isinstance(im, Image.Image) else Image.fromarray(np.asarray(im))
                diff = float(np.abs(np.asarray(im, np.float32) - np.asarray(img, np.float32))[mask > 0].mean())
                if diff < MIN_DE:
                    raise RuntimeError(f"edit left the organ unchanged ({diff:.2f}): {a.asset_id} {name} s{sd}")
                im.save(out)
                json.dump(dict(asset_id=a.asset_id, class_id=a.class_id, arm=name, prompt=pr, seed=sd, root_sha1=a.root_sha1,
                               mean_abs_rgb_change_in_mask=round(diff, 2), sec=round(time.time() - t, 1)), open(f"{out}.json", "w"), indent=1)
                print(a.asset_id, name, f"s{sd}", f"change {diff:.1f}", f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
