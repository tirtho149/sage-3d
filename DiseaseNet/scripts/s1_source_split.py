"""Stage 1: DEV accuracy split by source match — is the test photo's SAGE source among its class's parent sources?
Uses saved probe predictions (results/stage1/<subdir>/<SET>.npz, replicate 0). Recall per class is computed within
each stratum and macro-averaged over classes present in both strata. -> results/stage1/<subdir>/source_split.csv
usage: ~/Desktop/Research/.venv/bin/python scripts/s1_source_split.py <subdir> SET [SET ...]"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.s1 import harness as Hn  # noqa: E402
from src.generation.registry import assets  # noqa: E402
sub = sys.argv[1]; C = Hn.classes20(); D = Hn.dev_set(); D = D[D.class_id.isin(C)].reset_index(drop=True)
R = pd.read_parquet(f"{ROOT}/data/real_splits/real_splits.parquet", columns=["sha1", "class_id", "source"]).drop_duplicates(["sha1", "class_id"])
A = assets("stage0").merge(R, left_on=["root_sha1", "class_id"], right_on=["sha1", "class_id"], how="left")
psrc = A.groupby("class_id").source.agg(set).to_dict()
D["seen"] = [s in psrc.get(c, set()) for s, c in zip(D.source, D.class_id)]
rows = []
for name in sys.argv[2:]:
    z = np.load(f"{ROOT}/results/stage1/{sub}/{name}.npz"); y = z["dev_y"]; p = z["dev_proba"].argmax(1)
    for st in (True, False):
        m = D.seen.values == st
        rec = [np.mean(p[m & (y == k)] == k) for k in range(len(C)) if (m & (y == k)).sum() >= 5 and ((~m if st else D.seen.values) & (y == k)).sum() >= 5]
        rows.append(dict(set=name, source_seen=st, n=int(m.sum()), n_classes=len(rec), macro_recall=float(np.mean(rec)), acc=float(np.mean(p[m] == y[m]))))
T = pd.DataFrame(rows); T.to_csv(f"{ROOT}/results/stage1/{sub}/source_split.csv", index=False)
print(f"DEV photos from a parent source: {int(D.seen.sum())}/{len(D)}"); print(T.round(3).to_string(index=False))
