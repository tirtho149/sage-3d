"""Automated technical QC for every edited asset (brief s20). Writes results/statistics/qc_flags.csv.
Flags: missing/empty output, NaN/Inf parameters, negligible geometry change (geometry arms), exploding deformation,
negligible appearance change (appearance arms), silhouette collapse/disappearance, grain, fuzzy edges, inconsistent
render sizes, failed/unverified GE optimisation, unsupported geometric/appearance attributes, specimen-level camera
misalignment (alignment A3* IoU). Technical QC is separate from biological validity (not assessed: no experts)."""
import os, sys, json, glob, csv
import numpy as np
from plyfile import PlyData
from PIL import Image
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, f"{R}/src")
from metrics.kb_grounding import score as kb_geo
from rendering.rig_render import load_view
T = json.load(open(f"{R}/configs/evaluation/qc_thresholds.json"))
GEOM_ARMS, APP_ARMS = {"J", "G", "H"}, {"E", "F", "K", "H"}
KEYS = lambda sid: "Citrus__Canker__leaf" if sid == "Citrus__Canker__0004" else "__".join(sid.split("__")[:2])
ALIGN = {r["specimen"]: r for r in json.load(open(f"{R}/results/alignment/summary.json"))} if os.path.exists(f"{R}/results/alignment/summary.json") else {}


def qc_one(sid, arm, d):
    f = []; L = lambda p: json.load(open(p)) if os.path.exists(p) else None
    if not os.path.exists(f"{d}/asset.ply"): return ["missing_output"]
    v = PlyData.read(f"{d}/asset.ply")["vertex"].data
    if len(v) == 0: return ["empty_output"]
    bad = [k for k in v.dtype.names if not np.isfinite(v[k]).all()]
    if bad: f.append("nonfinite_params:" + "+".join(bad))
    gd, vm, ga, ak = L(f"{d}/gaussian_diff.json"), L(f"{d}/view_metrics.json"), L(f"{d}/geom_attributes.json"), L(f"{d}/appearance_kb.json")
    if gd:
        p95 = gd["disp_rel"]["p95"]
        if arm in GEOM_ARMS and p95 < T["disp_p95_min_geometry_arm"]: f.append("negligible_geometry_change")
        if p95 > T["disp_p95_max"]: f.append("exploding_deformation")
    if vm:
        s = vm["summary"]
        if arm in APP_ARMS and s["dE_obj"] < T["dE_min_appearance_arm"]: f.append("negligible_appearance_change")
        if s["grain_ratio"] > T["grain_ratio_max"]: f.append("grainy_texture")
        if s["fuzz_ratio"] > T["fuzz_ratio_max"]: f.append("fuzzy_edges")
        D = f"{R}/results/phaseA/{sid}/D_render"; areas = []
        for i in range(18):
            try:
                _, ad, _ = load_view(D, i); _, ax, _ = load_view(f"{d}/render", i); areas.append((ax > 0.5).sum() / max(1, (ad > 0.5).sum()))
            except FileNotFoundError: f.append("missing_render_views"); break
        if areas and min(areas) < T["silhouette_area_ratio_min"]: f.append("silhouette_collapse")
        sizes = {Image.open(p).size for p in glob.glob(f"{d}/render/rgb/*.png")}
        if len(sizes) > 1: f.append("inconsistent_render_sizes")
    else:
        f.append("no_render_metrics")
    if arm in ("F", "K", "H", "E"):
        ver = glob.glob(f"{d}/_ge/edit-n2n/*/verify.json") + glob.glob(f"{R}/results/pilot/{sid}/{arm}/_ge/edit-n2n/*/verify.json")
        vv = [json.load(open(x)) for x in ver]
        if not vv: f.append("ge_unverified(pre-fix run or log pruned)")
        elif not all(x.get("prompt_delivered") and x.get("last_ply") for x in vv): f.append("ge_verification_failed")
    if ga:
        g = kb_geo(ga, KEYS(sid))
        if g["unsupported_rate"] > 0: f.append("unsupported_geometry:" + "+".join(sorted(set(g["detected"]) - set(g["required"]))))
    if ak and ak.get("unsupported_appearance_rate", 0) > 0: f.append("unsupported_appearance:" + "+".join(sorted(set(ak["detected"]) - set(ak["required"]))))
    a = ALIGN.get(sid)
    if a and a["A3s_iou"] < T["alignment_iou_min"]: f.append("specimen_reconstruction_failure")
    return f


def main():
    rows = []
    for d in sorted(x for x in glob.glob(f"{R}/results/phaseA/*/*/seed*") if os.path.isdir(x)):
        sid, arm, seed = d.split("/")[-3], d.split("/")[-2], d.split("/")[-1]
        if arm in ("C_stage", "I"): continue
        fl = qc_one(sid, arm, d); rows.append(dict(sid=sid, arm=arm, seed=seed, n_flags=len(fl), flags=";".join(fl)))
    os.makedirs(f"{R}/results/statistics", exist_ok=True)
    with open(f"{R}/results/statistics/qc_flags.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["sid", "arm", "seed", "n_flags", "flags"]); w.writeheader(); [w.writerow(r) for r in rows]
    for r in rows: print(f"{r['sid']:26s} {r['arm']:2s} {r['seed']:6s} {r['flags'] or 'PASS'}")


if __name__ == "__main__":
    main()
