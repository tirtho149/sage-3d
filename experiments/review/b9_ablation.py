"""Review item B9: ablate the regularisation terms of the geometry editor on the curl operator.

Editor objective (src/geometry_editor/deform.py), written as the review's constrained form:
  G* = argmin  l_KB L_edit + l_anat L_arap + l_reg (L_smooth + L_anchor)   s.t. L_obs <= tau
L_edit = KB target (curl.curl_target), L_arap = edge-length preservation (tissue does not
stretch), L_smooth = Laplacian smoothness of the displacement, L_anchor = non-target parts
stay; L_obs = fitted input-view IoU drop, enforced as a hard check (evidence_policy, B2/B3).
Ablation (Mango 0049, Peach 0087; curl up, seed 0, 1000 iterations, same target):
  full: arap 1, smooth 0.01 | no_arap: arap 0 | no_smooth: smooth 0 | none: arap 0, smooth 0
Scored with the operator-independent measures of B1 (strain, non-rigidity, fine relief,
self-contact, midrib, roll change, fitted input-view IoU).

usage (sam3d env): CUDA_VISIBLE_DEVICES=0 LIDRA_SKIP_INIT=true ~/micromamba/envs/sam3d/bin/python experiments/review/b9_ablation.py
  -> results/review/b9_ablation/summary.{json,md}
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/experiments/review")
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
from geometry_editor.anatomy import leaf_frame, load_vertices, xyz  # noqa: E402
from geometry_editor.curl import curl_target, parse  # noqa: E402
from geometry_editor.deform import run as deform  # noqa: E402
from metrics.structure3d import input_view_iou  # noqa: E402
from b1_curl_robustness import evaluate  # noqa: E402

OUT = f"{ROOT}/results/review/b9_ablation"
LEAVES = ["Mango__Anthracnose__0049", "Peach__Leaf_Curl__0087"]
VARIANTS = {"full": dict(arap=1.0, smooth=0.01), "no_arap": dict(arap=0.0, smooth=0.01),
            "no_smooth": dict(arap=1.0, smooth=0.0), "none": dict(arap=0.0, smooth=0.0)}


def main():
    os.makedirs(OUT, exist_ok=True)
    P = parse("Leaves curl upwards and fall off."); res = {}
    for sid in LEAVES:
        F = leaf_frame(sid); base = evaluate(F, F["X"], None); base_iou = input_view_iou(F["ply"], sid)[0]
        res[sid] = dict(D=dict(input_view_iou=base_iou, self_contact_frac=base["self_contact_frac"]))
        tgt = curl_target(F, P["direction"], P["theta_deg"], P["r0"])
        for name, lam in VARIANTS.items():
            d = f"{OUT}/{sid}/{name}"; os.makedirs(d, exist_ok=True)
            np.save(f"{d}/t.npy", tgt.astype(np.float32)); np.save(f"{d}/n.npy", F["n"].astype(np.float32))
            if not os.path.exists(f"{d}/asset.ply"):
                spec = dict(seed=0, iters=1000, mode="sheet", normals_npy=f"{d}/n.npy",
                            ops=[dict(op="vector", target_npy=f"{d}/t.npy", kb_term="curl up")],
                            lambdas=dict(edit=1.0, arap=lam["arap"], smooth=lam["smooth"], anchor=0.0),
                            arap_coarse=20.0 if lam["arap"] > 0 else 0.0)
                deform(F["ply"], spec, f"{d}/asset.ply")
            Y = xyz(load_vertices(f"{d}/asset.ply")); m = evaluate(F, Y, base["roll_profile"])
            ch = np.array(m.get("roll_change_profile", []))
            keep = dict(median_roll_change=float(np.median(ch)) if len(ch) else None,
                        strain_02L_p95=m["strain_02L_p95"], nonrigidity_02L_p95=m["nonrigidity_02L_p95"],
                        fine_relief_corr=m["fine_relief_corr"],
                        self_contact_added=m["self_contact_frac"] - base["self_contact_frac"],
                        midrib_length_ratio=m["midrib_length_ratio"], input_view_iou=input_view_iou(f"{d}/asset.ply", sid)[0])
            res[sid][name] = keep
            print(sid, name, {k: round(v, 3) for k, v in keep.items() if v is not None}, flush=True)
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
    cols = ["median_roll_change", "strain_02L_p95", "nonrigidity_02L_p95", "fine_relief_corr", "self_contact_added",
            "midrib_length_ratio", "input_view_iou"]
    L = ["# B9: ablation of the editor's regularisation terms (curl up, seed 0)", "",
         "| leaf | variant | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for sid, r in res.items():
        for name in VARIANTS:
            m = r[name]; L.append(f"| {sid} | {name} | " + " | ".join(f"{m[c]:.3f}" if m[c] is not None else "-" for c in cols) + " |")
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
