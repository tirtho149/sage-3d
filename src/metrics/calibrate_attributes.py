"""Calibrate attribute thresholds and the operator x detector cross-talk matrix on a synthetic flat sheet.
Writes configs/evaluation/attribute_thresholds.json and results/geometry_calibration.json. Run in sam3d env."""
import sys, json, os, tempfile
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, f"{R}/src"); sys.path.insert(0, f"{R}/tests")
from test_geometry_ops import plane
from geometry_editor.deform import run
from metrics.geom_attributes import attributes
D = json.load(open(f"{R}/configs/geometry/operator_defaults.json")); T = tempfile.mkdtemp(); p = f"{T}/plane.ply"; plane(p)
KEY = {"distort": "distorted", "pucker": "puckered", "thicken": "thickened", "outgrowth": "outgrowth", "depress": "depression"}
resp = {}
for op in KEY:
    o = f"{T}/{op}.ply"; run(p, dict(seed=0, mode="sheet", ops=[dict(D[op], op=op)], iters=200), o); resp[op] = attributes(p, o)
thr = {"_doc": "Uniform rule on a synthetic flat sheet (src/metrics/calibrate_attributes.py; never tuned on specimens): "
       "threshold_k = midpoint between operator k's own response and the largest response of any OTHER operator on detector k. "
       "If the own response does not exceed every off-target response the detector is NON-SPECIFIC and is excluded from binary "
       "recall/unsupported-rate (raw score still reported). Pucker uses blister DENSITY (distinct blisters per (0.1 L)^2 of sheet area); the absolute count used first was not scale-invariant (deviation logged in reports/03)."}
spec = {}
for op, k in KEY.items():
    kk = "blister_density" if k == "puckered" else k
    own = resp[op][kk]; off = max(resp[o][kk] for o in KEY if o != op)
    spec[k] = bool(own > off); thr[k] = dict(score=kk, threshold=round((own + off) / 2, 5), own=round(own, 5), max_off_target=round(off, 5), specific=spec[k])
present = lambda a: {k: bool(thr[k]["specific"] and a[thr[k]["score"]] >= thr[k]["threshold"]) for k in KEY.values()}
cross = {op: present(resp[op]) for op in KEY}
json.dump(thr, open(f"{R}/configs/evaluation/attribute_thresholds.json", "w"), indent=1)
json.dump(dict(responses=resp, thresholds=thr, crosstalk=cross), open(f"{R}/results/geometry_calibration.json", "w"), indent=1)
print("operator -> detected attributes (diagonal should be True, off-diagonal False)")
for op in KEY: print(f"  {op:10s}", {k: int(v) for k, v in cross[op].items()})
