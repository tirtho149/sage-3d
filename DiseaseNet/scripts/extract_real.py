"""Extract real SAGE images of one partition for the Stage-0 classes into a local JPEG cache.

The locked TEST partition is only ever used for evaluation (never selection, never generation).
Cap per class (seeded) to bound evaluation cost; the cap and seed are recorded in the manifest.
usage: ~/Desktop/Research/.venv/bin/python scripts/extract_real.py <partition> [cap]
  -> data/real_splits/cache/<partition>/<sha1>.jpg, data/real_splits/stage0_<partition>.csv
"""
import io
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.provenance.leakage import Guard  # noqa: E402

SAGE = os.path.expanduser("~/Desktop/Research/data/raw/SAGE/data")
SEED = 20261015


def main(part, cap=300):
    Guard()                                            # verifies the lock before reading
    classes = [c["class_id"] for c in json.load(open(f"{ROOT}/configs/datasets/stage0_classes.json"))["classes"]]
    S = pd.read_parquet(f"{ROOT}/data/real_splits/real_splits.parquet")
    S = S[S.class_id.isin(classes) & (S.partition == part)].drop_duplicates("sha1")
    sel = pd.concat([g.sample(min(len(g), cap), random_state=SEED) for _, g in S.groupby("class_id")])
    out = f"{ROOT}/data/real_splits/cache/{part}"; os.makedirs(out, exist_ok=True)
    rows = []
    for shard, g in sel.groupby("shard"):
        pf = pq.ParquetFile(f"{SAGE}/{shard}")
        for rg, h in g.groupby("row_group"):
            col = pf.read_row_group(int(rg), columns=["image"]).column("image")
            for _, r in h.iterrows():
                p = f"{out}/{r.sha1}.jpg"
                if not os.path.exists(p):
                    im = Image.open(io.BytesIO(col[int(r.row_in_group)].as_py()["bytes"])).convert("RGB")
                    im.thumbnail((512, 512)); im.save(p, quality=92)
                rows.append(dict(path=os.path.relpath(p, ROOT), sha1=r.sha1, class_id=r.class_id, source=r.source, group=r.group))
    M = pd.DataFrame(rows); M["cap"] = cap; M["seed"] = SEED
    M.to_csv(f"{ROOT}/data/real_splits/stage0_{part}.csv", index=False)
    print(M.groupby("class_id").size().to_string()); print("total", len(M))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
