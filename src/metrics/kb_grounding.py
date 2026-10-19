"""KB-grounding metrics for 3D edits (brief s13), geometry part.
  geometry_recall      = |detected ∩ required| / |required|       required = KB operators of the selected stage
  unsupported_rate     = |detected \\ required| / |detected|        detected = specific detectors above threshold
Only detectors flagged specific on the synthetic calibration are used (configs/evaluation/attribute_thresholds.json).
usage: kb_grounding.py <geom_attributes.json> <class_key>   (prints JSON)
"""
import json, sys, os
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OP2ATTR = {"distort": "distorted", "pucker": "puckered", "thicken": "thickened", "outgrowth": "outgrowth", "depress": "depression"}


def score(attrs, key, thr=None, specs=None):
    thr = thr or json.load(open(f"{R}/configs/evaluation/attribute_thresholds.json"))
    specs = specs or json.load(open(f"{R}/data/kb/structured_specs.json"))
    req_all = {OP2ATTR[g["op"]] for g in specs[key]["geometry"]["operators"]}
    usable = {k for k, v in thr.items() if k != "_doc" and v["specific"]}
    det = {k for k in usable if attrs[thr[k]["score"]] >= thr[k]["threshold"]}
    req = req_all & usable
    return dict(required=sorted(req_all), required_scorable=sorted(req), not_scorable=sorted(req_all - usable), detected=sorted(det),
                geometry_recall=(len(det & req) / len(req)) if req else None,
                unsupported_rate=(len(det - req_all) / len(det)) if det else 0.0)


if __name__ == "__main__":
    print(json.dumps(score(json.load(open(sys.argv[1])), sys.argv[2])))
