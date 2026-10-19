"""2D-arm metrics for all specimens -> results/statistics/metrics2d.csv and per-arm summary (specimen-averaged)."""
import os, sys, glob, csv, json
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, f"{R}/src")
from metrics.edit2d_metrics import metrics
from metrics.appearance_kb import main2d
KEY = lambda sid: "Citrus__Canker__leaf" if sid == "Citrus__Canker__0004" else "__".join(sid.split("__")[:2])
rows = []
for I in sorted(glob.glob(f"{R}/data/inputs/*__*__*")):
    sid = os.path.basename(I)
    if not os.path.isdir(f"{R}/results/pilot/{sid}") and not os.path.isdir(f"{R}/results/phaseA/{sid}"): continue
    for arm in ("B", "C", "C_nogeo", "C_mismatch", "C_stage", "I"):
        for f in sorted(glob.glob(f"{R}/results/pilot/{sid}/{arm}/seed*.png") + glob.glob(f"{R}/results/phaseA/{sid}/{arm}/seed*.png")):
            m = metrics(f"{I}/image.png", f"{I}/0.png", f); a = main2d(f"{I}/image.png", f"{I}/0.png", f, KEY(sid))
            rows.append(dict(sid=sid, arm=arm, seed=os.path.basename(f)[4:-4], **m, appearance_recall=a["appearance_recall"],
                             unsupported_appearance_rate=a["unsupported_appearance_rate"], detected=";".join(a["detected"])))
os.makedirs(f"{R}/results/statistics", exist_ok=True)
with open(f"{R}/results/statistics/metrics2d.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
summ = {}
for arm in ("B", "C", "C_nogeo", "C_mismatch", "C_stage", "I"):
    per = {}
    for r in rows:
        if r["arm"] == arm: per.setdefault(r["sid"], []).append(r)
    if not per: continue
    agg = lambda k: [np.mean([x[k] for x in v if x[k] is not None]) for v in per.values() if any(x[k] is not None for x in v)]
    summ[arm] = {k: (float(np.mean(agg(k))) if agg(k) else None) for k in ("dE_in", "outside_change", "detail_ratio", "bg_change", "appearance_recall", "unsupported_appearance_rate")}
    summ[arm]["n_specimens"] = len(per)
json.dump(summ, open(f"{R}/results/statistics/metrics2d_summary.json", "w"), indent=1)
for a, v in summ.items(): print(a, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()})
