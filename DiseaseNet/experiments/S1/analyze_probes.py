"""Stage 1: paired contrasts between probe runs, pooled over matched replicates, from saved per-image predictions.

For contrast A - B on split S (dev/final): per replicate r the macro-F1 difference on the same test images; the
pooled estimate is the mean over replicates present for BOTH arms; 95% CI and two-sided p from a group bootstrap in
which every resample of near-duplicate groups is applied to all replicates (2,000 resamples). Holm adjustment is
applied to the contrasts flagged primary.
usage: ~/Desktop/Research/.venv/bin/python experiments/S1/analyze_probes.py <subdir> <contrast-spec.json> [split]
  spec: [{"name":..., "a":SET, "b":SET, "primary":bool, "classes": optional list}, ...]
  -> results/stage1/<subdir>/contrasts_<spec-stem>_<split>.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from src.evaluation import metrics as Mx  # noqa: E402
from src.s1.harness import holm  # noqa: E402


def load(d, name, r, split):
    p = f"{d}/{name}{'' if r == 0 else f'~{r}'}.npz"
    return np.load(p) if os.path.exists(p) else None


def pooled(d, a, b, split="dev", n=2000, seed=0, reps=range(10), keep=None):
    A = {r: load(d, a, r, split) for r in reps}; B = {r: load(d, b, r, split) for r in reps}
    R = [r for r in reps if A[r] is not None and B[r] is not None]
    if not R:
        return None
    y, g = A[R[0]][f"{split}_y"], A[R[0]][f"{split}_g"]
    k = A[R[0]][f"{split}_proba"].shape[1]
    sel = np.arange(len(y)) if keep is None else np.flatnonzero(np.isin(y, keep))
    labels = list(range(k)) if keep is None else list(keep)
    pa = {r: A[r][f"{split}_proba"].argmax(1) for r in R}; pb = {r: B[r][f"{split}_proba"].argmax(1) for r in R}
    f = lambda yy, pp: Mx.summary(yy, pp, classes=labels)["macro_f1"]  # noqa: E731
    est = [f(y[sel], pa[r][sel]) - f(y[sel], pb[r][sel]) for r in R]
    ys, gs = y[sel], g[sel]; ug = np.unique(gs); idx = {x: np.flatnonzero(gs == x) for x in ug}
    rng = np.random.default_rng(seed); dist = []
    for _ in range(n):
        s = np.concatenate([idx[x] for x in rng.choice(ug, len(ug), replace=True)])
        dist.append(np.mean([f(ys[s], pa[r][sel][s]) - f(ys[s], pb[r][sel][s]) for r in R]))
    dist = np.array(dist)
    return dict(a=a, b=b, reps=len(R), diff=float(np.mean(est)), per_rep=[round(x, 4) for x in est],
                ci_lo=float(np.percentile(dist, 2.5)), ci_hi=float(np.percentile(dist, 97.5)),
                p=float(min(1.0, 2 * min((dist <= 0).mean(), (dist >= 0).mean()))),
                a_f1=float(np.mean([f(y[sel], pa[r][sel]) for r in R])), b_f1=float(np.mean([f(y[sel], pb[r][sel]) for r in R])))


def main():
    sub, spec = sys.argv[1], sys.argv[2]; split = sys.argv[3] if len(sys.argv) > 3 else "dev"
    d = f"{ROOT}/results/stage1/{sub}"; rows = []
    for c in json.load(open(spec)):
        r = pooled(d, c["a"], c["b"], split, keep=c.get("keep"))
        if r is None:
            print("missing", c["name"]); continue
        r.update(name=c["name"], primary=bool(c.get("primary"))); rows.append(r)
        print(f"{c['name']:34s} {r['a_f1']:.3f} vs {r['b_f1']:.3f}  diff {r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] p={r['p']:.3f} reps={r['reps']}", flush=True)
    D = pd.DataFrame(rows)
    if len(D) and D.primary.any():
        adj = holm(dict(zip(D[D.primary].name, D[D.primary].p))); D["p_holm"] = D.name.map(adj)
    D.to_csv(f"{d}/contrasts_{os.path.splitext(os.path.basename(spec))[0]}_{split}.csv", index=False)


if __name__ == "__main__":
    main()
