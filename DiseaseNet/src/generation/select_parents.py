"""Stage 0 step 1: choose parent photos and make organ masks for new 3D assets.

For every Stage-0 class without existing assets: draw candidate photos from the PARENT
partition only (Guard-checked), one per near-duplicate group, round-robin over source
datasets (diversity), >= 256 px on the short side. Mask each with SAM ViT-H from a
centred box (6% margin; the photographed organ is usually central in SAGE close-ups) and
accept a candidate when the mask passes QC:
   SAM IoU score >= 0.88, 5% <= coverage <= 85%, largest component >= 90% of the mask,
   mask touches at most 2 image borders (not the background).
The first N_ASSETS accepted candidates per class become parent assets; all decisions are
logged (data/synthetic/stage0/parents.csv). Pilot classes reuse the 3 SAGE-Extend assets.

usage (Research .venv, GPU 0): CUDA_VISIBLE_DEVICES=0 ~/Desktop/Research/.venv/bin/python -m src.generation.select_parents
Stage 1 (parent-diversity scaling): --stage1 --total 12 adds parents to data/synthetic/stage1/inputs until every
Stage-0 class (pilot classes included) has `total` parents counting its Stage-0 ones; excludes Stage-0 roots and their
near-duplicate groups and photos with more than one SAGE label; seed 20261016; same mask QC.
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.provenance.leakage import Guard  # noqa: E402

SAGE = os.path.expanduser("~/Desktop/Research/data/raw/SAGE/data")
OUT = f"{ROOT}/data/synthetic/stage0"
N_ASSETS, N_CAND, SEED = 3, 10, 20261015
MARGIN = 0.06
NEW_SEEDED = {"potato::tomato_spotted_wilt_virus", "strawberry::leaf_scorch", "strawberry::leaf_spot"}


def hash_seed(cid):
    import hashlib
    return int(hashlib.sha256(f"{SEED}:{cid}".encode()).hexdigest()[:8], 16)


PILOT = {"peach::leaf_curl", "tomato::leaf_mold", "corn::smut", "citrus::canker", "mango::anthracnose"}


def load_image(row):
    pf = pq.ParquetFile(f"{SAGE}/{row.shard}")
    col = pf.read_row_group(int(row.row_group), columns=["image"]).column("image")
    return Image.open(io.BytesIO(col[int(row.row_in_group)].as_py()["bytes"])).convert("RGB")


def qc(mask, score, img=None):
    cov = mask.mean()
    filled = ndimage.binary_fill_holes(mask)
    hole_ratio = filled.sum() / max(mask.sum(), 1)            # > 1.25: mask wraps a hole -> background selected
    bright = float(np.asarray(img, np.float64)[mask].mean() / 255) if img is not None and mask.any() else 1.0
    lab, n = ndimage.label(mask)
    big = np.bincount(lab.ravel())[1:].max() / max(mask.sum(), 1) if n else 0.0
    borders = sum([mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()])
    ok = (score >= 0.88 and 0.05 <= cov <= 0.85 and big >= 0.90 and borders <= 2
          and hole_ratio <= 1.25 and bright >= 0.15)
    return ok, dict(sam_score=float(score), coverage=float(cov), largest_cc=float(big), borders=int(borders),
                    hole_ratio=float(hole_ratio), brightness=float(bright))


def main():
    global OUT, N_ASSETS, SEED
    ap = argparse.ArgumentParser(); ap.add_argument("--stage1", action="store_true"); ap.add_argument("--total", type=int, default=12)
    args = ap.parse_args()
    from transformers import SamModel, SamProcessor
    have0 = {}
    if args.stage1:
        sys.path.insert(0, ROOT)
        from src.generation.registry import assets
        A0 = assets("stage0"); have0 = A0.groupby("class_id").size().to_dict(); roots0 = set(A0.root_sha1)
        OUT = f"{ROOT}/data/synthetic/stage1"; SEED = 20261016
    os.makedirs(OUT, exist_ok=True)
    classes = [c["class_id"] for c in json.load(open(f"{ROOT}/configs/datasets/stage0_classes.json"))["classes"]]
    S = pd.read_parquet(f"{ROOT}/data/real_splits/real_splits.parquet")
    M = pd.read_parquet(f"{ROOT}/data/manifests/sage_images.parquet",
                        columns=["sha1", "shard", "row_group", "row_in_group", "source", "filename", "class_id"])
    guard = Guard(); rng = np.random.default_rng(SEED)
    vr = f"{OUT}/visual_rejects.csv"                  # documented visual QC (agent review of the contact sheet)
    rejected = set(pd.read_csv(vr).sha1) if os.path.exists(vr) else set()
    dev = "cuda"; model = SamModel.from_pretrained("facebook/sam-vit-huge").to(dev).eval()
    proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    log = []
    multi = set(S.groupby("sha1").class_id.nunique().pipe(lambda x: x[x > 1]).index) if args.stage1 else set()
    for cid in classes:
        if cid in PILOT and not args.stage1:
            continue
        have = [d for d in os.listdir(f"{OUT}/inputs")] if os.path.isdir(f"{OUT}/inputs") else []
        n_have = sum(d.startswith(cid.replace("::", "__") + "__") for d in have)
        N_ASSETS = args.total - have0.get(cid, 0) if args.stage1 else N_ASSETS
        if n_have >= N_ASSETS:
            continue                                   # idempotent: keep earlier picks untouched
        if args.stage1:
            rng = np.random.default_rng(abs(hash_seed(cid)))
        elif cid in NEW_SEEDED:                        # classes added by amendment get their own seed
            rng = np.random.default_rng(abs(hash_seed(cid)))
        P = S[(S.class_id == cid) & (S.partition == "parent")].drop_duplicates("group")
        if args.stage1:
            g0 = set(S[S.sha1.isin(roots0)].group)
            P = P[~P.sha1.isin(roots0) & ~P.group.isin(g0) & ~P.sha1.isin(multi)]
        P = P.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        order, buckets = [], {s: list(g.index) for s, g in P.groupby("source")}   # round-robin over sources
        while any(buckets.values()) and len(order) < N_CAND * 6 * (4 if args.stage1 else 1):
            for s in list(buckets):
                if buckets[s]:
                    order.append(buckets[s].pop(0))
        accepted = n_have
        for idx in order:
            if accepted >= N_ASSETS:
                break
            r = P.loc[idx]
            if os.path.isdir(f"{OUT}/inputs/{cid.replace('::', '__')}__{r.sha1[:8]}"):
                continue
            if r.sha1 in rejected:
                log.append(dict(class_id=cid, sha1=r.sha1, source=r.source, filename=r.filename,
                                accepted=False, reason="visual QC reject")); continue
            guard.assert_generation_roots([r.sha1])
            im = load_image(r); W, H = im.size
            rec = dict(class_id=cid, sha1=r.sha1, source=r.source, filename=r.filename, w=W, h=H)
            if min(W, H) < 256:
                log.append({**rec, "accepted": False, "reason": "too small"}); continue
            box = [[[MARGIN * W, MARGIN * H, (1 - MARGIN) * W, (1 - MARGIN) * H]]]
            inp = proc(im, input_boxes=box, return_tensors="pt").to(dev)
            with torch.no_grad():
                o = model(**inp, multimask_output=True)
            masks = proc.image_processor.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                                            inp["reshaped_input_sizes"].cpu())[0][0]
            sc = o.iou_scores[0, 0].cpu().numpy(); k = int(sc.argmax()); m = masks[k].numpy().astype(bool)
            ok, q = qc(m, sc[k], im)
            rec.update(q)
            if not ok:
                log.append({**rec, "accepted": False, "reason": "mask QC"}); continue
            aid = f"{cid.replace('::', '__')}__{r.sha1[:8]}"
            d = f"{OUT}/inputs/{aid}"; os.makedirs(d, exist_ok=True)
            im.save(f"{d}/image.png"); Image.fromarray(m.astype(np.uint8) * 255).save(f"{d}/0.png")
            json.dump({**rec, "asset_id": aid, "root_sha1": [r.sha1], "box_margin": MARGIN,
                       "sam_model": "facebook/sam-vit-huge"}, open(f"{d}/mask_meta.json", "w"), indent=1)
            log.append({**rec, "accepted": True, "reason": "", "asset_id": aid}); accepted += 1
        print(f"{cid}: {accepted}/{N_ASSETS} accepted from {sum(1 for x in log if x['class_id'] == cid)} tried "
              f"({P.source.nunique()} sources in parent pool)", flush=True)
    prev = pd.read_csv(f"{OUT}/parents.csv") if os.path.exists(f"{OUT}/parents.csv") else pd.DataFrame()
    pd.concat([prev, pd.DataFrame(log)]).to_csv(f"{OUT}/parents.csv", index=False)


if __name__ == "__main__":
    main()
