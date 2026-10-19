"""Stage-0 asset registry: every parent asset with its class, root photo, input files and splat.

New assets: data/synthetic/<stage>/{inputs,assets}/<asset_id>/ (this project).
Pilot assets: the 15 SAGE-Extend reconstructions, referenced read-only (never rebuilt):
  inputs   ~/Desktop/Research/external/SAGE-Extend/data/inputs/<Crop__Disease__NNNN>/{image.png,0.png}
  splat    .../results/pilot/sam3d/<sid>/seed0/splat_ge.ply  (+ pose.npz with pointmap)
"""
import glob
import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.expanduser("~/Desktop/Research/external/SAGE-Extend")
PILOT_CLASS = {"Peach__Leaf_Curl": "peach::leaf_curl", "Tomato__Leaf_Mold": "tomato::leaf_mold", "Corn__Smut": "corn::smut",
               "Citrus__Canker": "citrus::canker", "Mango__Anthracnose": "mango::anthracnose"}


def assets(stage="stage0", include_pilot=True):
    """stage0: the 60 Stage-0 assets. s1all: Stage-0 assets + Stage-1 extra parents (data/synthetic/stage1/inputs)."""
    if stage == "s1all":
        return pd.concat([assets("stage0"), assets("stage1", include_pilot=False)], ignore_index=True)
    rows = []
    for d in sorted(glob.glob(f"{ROOT}/data/synthetic/{stage}/inputs/*")):
        m = json.load(open(f"{d}/mask_meta.json")); a = f"{ROOT}/data/synthetic/{stage}/assets/{m['asset_id']}"
        ok = os.path.exists(f"{a}/run_meta.json") and json.load(open(f"{a}/run_meta.json")).get("status") == "ok"
        rows.append(dict(asset_id=m["asset_id"], class_id=m["class_id"], root_sha1=m["root_sha1"][0], image=f"{d}/image.png",
                         mask=f"{d}/0.png", splat=f"{a}/splat.ply" if ok else None, run_dir=a if ok else None, origin="diseasenet"))
    if not include_pilot:
        return pd.DataFrame(rows)
    spec = pd.read_csv(f"{EXT}/data/metadata/specimen_manifest.csv").set_index("specimen_id")
    for d in sorted(glob.glob(f"{EXT}/results/pilot/sam3d/*__*")):
        sid = os.path.basename(d); key, num = sid.rsplit("__", 1)
        if key not in PILOT_CLASS:
            continue
        run = f"{d}/seed0"
        rows.append(dict(asset_id=f"pilot__{sid}", class_id=PILOT_CLASS[key], root_sha1=spec.loc[f"{key}/{num}", "sha1"],
                         image=f"{EXT}/data/inputs/{sid}/image.png", mask=f"{EXT}/data/inputs/{sid}/0.png",
                         splat=f"{run}/splat_ge.ply" if os.path.exists(f"{run}/splat_ge.ply") else None, run_dir=run, origin="sage-extend-pilot"))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    A = assets(); print(A.groupby("class_id").agg(n=("asset_id", "size"), ready=("splat", lambda s: s.notna().sum())).to_string())
