"""Within-crop disease discrimination (chance = 0.5): for each DEV photo, compare the probabilities of the two classes
of its crop and score whether the true one is higher; macro over classes; group-bootstrap 95% CI.
For the cross-class (knowledge-only) arms this is the direct test of whether the KB edit carries class information.
usage: python scripts/s1_within_crop.py <subdir> SET [SET ...] -> results/stage1/<subdir>/within_crop.csv"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.s1 import harness as Hn  # noqa: E402
sub = sys.argv[1]; d = f"{ROOT}/results/stage1/{sub}"; rows = []
for name in sys.argv[2:]:
    j = pd.read_json(f"{d}/{name}.json", typ="series"); z = np.load(f"{d}/{name}.npz")
    C = list(j["dev"]["per_class_recall"].keys()); crop = np.array([c.split("::")[0] for c in C])
    sib = np.array([[k for k in range(len(C)) if crop[k] == crop[i] and k != i][0] for i in range(len(C))])
    y, P, g = z["dev_y"], z["dev_proba"], z["dev_g"]
    ok = (P[np.arange(len(y)), y] > P[np.arange(len(y)), sib[y]]).astype(float)
    def stat(ix):
        return np.mean([ok[ix][y[ix] == k].mean() for k in range(len(C)) if (y[ix] == k).any()])
    ug = np.unique(g); idx = {x: np.flatnonzero(g == x) for x in ug}; rng = np.random.default_rng(0)
    bs = [stat(np.concatenate([idx[x] for x in rng.choice(ug, len(ug))])) for _ in range(1000)]
    rows.append(dict(set=name, within_crop_acc=stat(np.arange(len(y))), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5),
                     per_class=dict(zip(C, [round(ok[y == k].mean(), 3) for k in range(len(C))]))))
    print(f"{name:10s} within-crop {rows[-1]['within_crop_acc']:.3f} [{rows[-1]['lo']:.3f}, {rows[-1]['hi']:.3f}]", flush=True)
pd.DataFrame(rows).to_csv(f"{d}/within_crop.csv", index=False)
