"""Stage 1: frozen DINOv2 + logistic-regression probes (T0) for named training sets on DEV (+ synthetic LOPO).
Skips sets whose result exists or whose images are incomplete (every class needs >= 3 parents).
usage: CUDA_VISIBLE_DEVICES=0 ~/Desktop/Research/.venv/bin/python experiments/S1/run_probes.py <out_subdir> SET[~r] [SET[~r] ...]
  optional first arg cohort=x16 (cross-class cohort: corn and citrus excluded; 16-class head)
  SET~r = replicate r (default 0) -> results/stage1/<out_subdir>/<SET>~r.json/.npz (r=0 keeps the plain name)
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from src.s1 import harness as Hn  # noqa: E402
from src.s1.sets import make  # noqa: E402

out = f"{ROOT}/results/stage1/{sys.argv[1]}"; C = Hn.classes20(); names = sys.argv[2:]
if names and names[0].startswith("cohort="):          # cohort=x16: the 16 classes of the 8 same-organ crop pairs
    co = names.pop(0).split("=")[1]
    if co == "x16":
        C = [c for c in C if c.split("::")[0] not in ("corn", "citrus")]
    if co == "div18":
        from src.s1.sets import DIV_EXCLUDE
        C = [c for c in C if c not in DIV_EXCLUDE]
for name in names:
    if os.path.exists(f"{out}/{name}.json"):
        print("exists", name); continue
    try:
        base, rep = (name.split("~")[0], int(name.split("~")[1])) if "~" in name else (name, 0)
        S = make(base, rep)
    except FileNotFoundError as e:
        print("missing", name, e); continue
    S = S[S.class_id.isin(C)]; cov = S.groupby("class_id").root_sha1.nunique()
    if len(cov) < len(C) or cov.min() < 3:
        print("incomplete", name, f"{len(cov)} classes, min parents {cov.min() if len(cov) else 0}"); continue
    r = Hn.run_probe(name, S, C, out)
    print(f"{name:14s} n={r['n_train']} parents={r['n_parents']} LOPO {r['lopo_macro_f1']:.3f} DEV {r['dev']['macro_f1']:.3f} "
          f"[{r['dev']['macro_f1_ci'][0]:.3f}, {r['dev']['macro_f1_ci'][1]:.3f}]", flush=True)
