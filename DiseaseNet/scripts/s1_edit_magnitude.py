"""Stage 1 audit: how much do the Stage-0 2D edits change the parent photo? (CPU)
Per asset: CIEDE76 (Lab Euclidean) between the input photo and kb_s0 / generic_s0 inside the organ mask and outside it;
fraction of organ pixels with dE > 10. -> results/stage1/audit/edit_magnitude.csv + printed class summary."""
import glob, json, os
import numpy as np, pandas as pd
from PIL import Image
from skimage.color import rgb2lab
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ST = f"{ROOT}/data/synthetic/stage0"
rows = []
for d in sorted(glob.glob(f"{ST}/inputs/*/")):
    aid = os.path.basename(d[:-1]); img = Image.open(f"{d}image.png").convert("RGB"); m = np.asarray(Image.open(f"{d}0.png").convert("L").resize(img.size)) > 127
    L0 = rgb2lab(np.asarray(img) / 255.0)
    for k in ("kb", "generic"):
        p = f"{ST}/edits2d/{aid}/{k}_s0.png"
        if not os.path.exists(p): continue
        L1 = rgb2lab(np.asarray(Image.open(p).convert("RGB").resize(img.size)) / 255.0); dE = np.linalg.norm(L1 - L0, axis=2)
        rows.append(dict(asset_id=aid, class_id=aid.split("__")[0] + "::" + aid.split("__")[1], edit=k, dE_in=float(dE[m].mean()),
                         dE_out=float(dE[~m].mean()) if (~m).any() else np.nan, frac_gt10=float((dE[m] > 10).mean())))
D = pd.DataFrame(rows); os.makedirs(f"{ROOT}/results/stage1/audit", exist_ok=True); D.to_csv(f"{ROOT}/results/stage1/audit/edit_magnitude.csv", index=False)
print(D.groupby("edit")[["dE_in", "dE_out", "frac_gt10"]].describe().T.round(2).to_string())
print(D[D.edit == "kb"].groupby("class_id")[["dE_in", "frac_gt10"]].mean().round(2).to_string())
