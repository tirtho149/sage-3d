"""Phase A/B geometry arms for one specimen (sam3d env, GPU optional).
J = KB geometry-only (operators from data/kb/structured_specs.json), G = generic smooth field, RMS-matched to J.
Each output: <arm>/seed<k>/asset.ply (+ .json report) and rig renders (rgb/alpha/depth) from the D rig.
usage: run_geometry.py <specimen_id e.g. Peach__Leaf_Curl__0087> <class_key> [seeds=0,1,2]
"""
import os, sys, json, subprocess
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, f"{R}/src")
from geometry_editor.deform import run as deform
from rendering.rig_render import render_rig, save_view
from PIL import Image

sid, key = sys.argv[1], sys.argv[2]; seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "0,1,2").split(",")]
S = json.load(open(f"{R}/data/kb/structured_specs.json"))[key]
src = f"{R}/results/pilot/sam3d/{sid}/seed0/splat_ge.ply"; rig = json.load(open(f"{R}/results/pilot/{sid}/D/rig.json"))
OUT = f"{R}/results/phaseA/{sid}"
DEFAULTS = json.load(open(f"{R}/configs/geometry/operator_defaults.json"))   # per-operator parameters (documented)


def save_renders(ply, d):
    for k in ("rgb", "alpha", "depth"): os.makedirs(f"{d}/{k}", exist_ok=True)
    for r in render_rig(ply, rig):
        save_view(d, r)


log = []
if not os.path.exists(f"{OUT}/D_render/rgb/017.png"): save_renders(f"{R}/results/pilot/sam3d/{sid}/seed0/splat.ply", f"{OUT}/D_render")
ops = [dict(DEFAULTS[g["op"]], op=g["op"], kb_term=g["term"], kb_phrase=g["phrase"]) for g in S["geometry"]["operators"]]
for seed in seeds:
    if not ops:
        log.append(dict(arm="J", seed=seed, status="ABSTAIN", reason="no implemented, unhedged geometric operator in the selected KB stage",
                        unsupported=S["geometry"]["unsupported_operators"])); continue
    d = f"{OUT}/J/seed{seed}"; os.makedirs(d, exist_ok=True)
    if not os.path.exists(f"{d}/asset.ply"):
        rJ = deform(src, dict(seed=seed, ops=ops, iters=300), f"{d}/asset.ply"); save_renders(f"{d}/asset.ply", f"{d}/render")
    rJ = json.load(open(f"{d}/asset.ply.json"))
    # G: generic smooth vector field with the same RMS displacement as J (no KB, no normal constraint)
    from plyfile import PlyData
    a = PlyData.read(src)["vertex"]; b = PlyData.read(f"{d}/asset.ply")["vertex"]
    disp = np.stack([b[k] - a[k] for k in "xyz"], 1); rms = float(np.sqrt((disp ** 2).sum(1).mean()) / rJ["L_bbox_diag"])
    g = f"{OUT}/G/seed{seed}"; os.makedirs(g, exist_ok=True)
    if not os.path.exists(f"{g}/asset.ply"):
        deform(src, dict(seed=seed, ops=[dict(op="generic", rms=rms, n=6, sigma=0.15)], iters=300), f"{g}/asset.ply"); save_renders(f"{g}/asset.ply", f"{g}/render")
    log.append(dict(arm="J+G", seed=seed, status="OK", J_disp_rms_rel=rms))
json.dump(log, open(f"{OUT}/geometry_log.json", "w"), indent=1); print(json.dumps(log))
