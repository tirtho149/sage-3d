"""Sibling map for the cross-class arms (light module: no diffusion imports)."""
import json
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_p = {}
for _c in json.load(open(f"{ROOT}/configs/datasets/stage0_classes.json"))["classes"]:
    _p.setdefault(_c["class_id"].split("::")[0], []).append(_c["class_id"])
SIBLING = {a: b for pair in _p.values() if len(pair) == 2 for a, b in (pair, pair[::-1])}
