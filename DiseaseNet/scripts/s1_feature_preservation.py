"""Stage 1, step 3C: does a 3D arm preserve the parent photo's diagnostic appearance at the INPUT view? (GPU for features)

Per asset and 3D arm, compare the input-view render on white (y+0p+0, R0, replicate 0) with the parent photo on white
(PHOTO2D v0 R0): DINOv2 ViT-S/14 cosine; high-frequency energy ratio (variance of the Laplacian of L* inside the
object, render / photo; < 1 = fine texture lost); mean colour difference (CIEDE76 between object-mean Lab colours);
object IoU is taken from the generator (input_iou). Registration is approximate (both crops are object-centred
squares), so pixel metrics such as LPIPS are not used. Perceptual similarity is NOT evidence of biological correctness.
Outputs results/stage1/feature_preservation/{per_asset.csv, summary.csv}
usage: CUDA_VISIBLE_DEVICES=0 ~/Desktop/Research/.venv/bin/python scripts/s1_feature_preservation.py
"""
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.s1.harness import manifest  # noqa: E402
from src.training.features import extract  # noqa: E402

ARMS = ["U3D", "HARM3D", "TEX3D", "GEO3D", "TEXGEO3D", "GEN3D", "KBAPP3D", "KB3D"]


def obj_stats(path):
    x = np.asarray(Image.open(os.path.join(ROOT, path)).convert("RGB")).astype(np.float32) / 255
    m = ~(x.min(2) > 0.97)                                    # non-white = object (R0 renders/photos are on white)
    m = ndimage.binary_erosion(m, iterations=2)
    L = rgb2lab(x)
    hf = ndimage.laplace(L[..., 0])[m].var() if m.sum() > 50 else np.nan
    return hf, L[m].mean(0) if m.sum() > 50 else np.full(3, np.nan)


def main():
    P = manifest("PHOTO2D"); P = P[(P.view == "v0") & (P.cond == "R0") & (P.edit_seed == 0)].set_index("asset_id")
    rows = []
    for arm in ARMS:
        try:
            M = manifest(arm)
        except FileNotFoundError:
            print("missing", arm); continue
        M = M[(M.view == "y+0p+0") & (M.cond == "R0") & (M.edit_seed == 0)]
        for _, r in M.iterrows():
            if r.asset_id in P.index:
                rows.append(dict(arm=arm, asset_id=r.asset_id, class_id=r.class_id, render=r.path, photo=P.loc[r.asset_id, "path"],
                                 input_iou=r.input_iou, geo_ops=r.geo_ops if isinstance(r.geo_ops, str) else ""))
    D = pd.DataFrame(rows)
    Fr = extract(D.render.tolist()); Fp = extract(D.photo.tolist())
    D["dino_cos"] = (Fr * Fp).sum(1) / (np.linalg.norm(Fr, axis=1) * np.linalg.norm(Fp, axis=1))
    hr, dE = [], []
    for _, r in D.iterrows():
        h1, c1 = obj_stats(r.render); h0, c0 = obj_stats(r.photo)
        hr.append(h1 / h0 if h0 and h0 > 0 else np.nan); dE.append(float(np.linalg.norm(c1 - c0)))
    D["hf_ratio"] = hr; D["colour_dE"] = dE
    out = f"{ROOT}/results/stage1/feature_preservation"; os.makedirs(out, exist_ok=True)
    D.to_csv(f"{out}/per_asset.csv", index=False)
    S = D.groupby("arm")[["dino_cos", "hf_ratio", "colour_dE", "input_iou"]].median().round(3).sort_values("dino_cos")
    S.to_csv(f"{out}/summary.csv"); print(S.to_string())


if __name__ == "__main__":
    main()
