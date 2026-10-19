"""Modality x generator interaction: [A - B on the geometry cohort] - [A - B on the appearance cohort], pooled over matched
replicates; each group-bootstrap resample is applied to both cohorts (2,000 resamples).
usage: python scripts/s1_interaction.py A B [subdir] -> printed + results/stage1/<subdir>/interaction_<A>_<B>.json"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.evaluation import metrics as Mx  # noqa: E402
from src.s1.harness import classes20  # noqa: E402
a, b = sys.argv[1], sys.argv[2]; sub = sys.argv[3] if len(sys.argv) > 3 else "stage0r"; d = f"{ROOT}/results/stage1/{sub}"
C = classes20(); G = [C.index(c) for c in ["peach::leaf_curl", "corn::smut", "mango::anthracnose", "citrus::greening_disease"]]
A_ = [i for i in range(len(C)) if i not in G]
def ld(n, r): return np.load(f"{d}/{n}{'' if r == 0 else f'~{r}'}.npz")
R = [r for r in range(3) if os.path.exists(f"{d}/{a}{'' if r == 0 else f'~{r}'}.npz") and os.path.exists(f"{d}/{b}{'' if r == 0 else f'~{r}'}.npz")]
z0 = ld(a, R[0]); y, g = z0["dev_y"], z0["dev_g"]
pa = {r: ld(a, r)["dev_proba"].argmax(1) for r in R}; pb = {r: ld(b, r)["dev_proba"].argmax(1) for r in R}
def f(ix, labels, p): return Mx.summary(y[ix], p[ix], classes=labels)["macro_f1"]
def inter(ix):
    gi, ai = ix[np.isin(y[ix], G)], ix[np.isin(y[ix], A_)]
    return np.mean([(f(gi, G, pa[r]) - f(gi, G, pb[r])) - (f(ai, A_, pa[r]) - f(ai, A_, pb[r])) for r in R])
est = inter(np.arange(len(y))); ug = np.unique(g); idx = {x: np.flatnonzero(g == x) for x in ug}; rng = np.random.default_rng(0)
bs = np.array([inter(np.concatenate([idx[x] for x in rng.choice(ug, len(ug))])) for _ in range(2000)])
out = dict(a=a, b=b, reps=len(R), interaction=float(est), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
           p=float(min(1, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))))
json.dump(out, open(f"{d}/interaction_{a}_{b}.json", "w"), indent=1); print(out)
