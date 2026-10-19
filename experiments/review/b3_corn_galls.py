"""Review item B3: corn smut galls -- preserve-mode placement, severity sweep, scale-free reporting.

Arms (Corn 0012 and 0052; 0025 abstains, no kernel tissue), seeds 0-2:
  V   KB placement on kernel tissue (Test 2 demo, reused from results/demos/corn)
  Vr  same galls, random sites on the whole ear (matched generic placement; reused)
  Vh  preserve mode: kernel tissue NOT visible from the input camera (new)
  Vh_n{1,3,6}  severity sweep in preserve mode: 1, 3, 6 galls (new; n=3 is Vh)
Measures (all scale-free; nothing uses the assumed 18 cm):
  added volume / L_ear^3 (per mille), gall equivalent diameter / L_ear, fitted input-view IoU
  and the evidence-policy decision (TAU from b2), fraction of kernel Gaussians hidden by galls
  (the rest of the ear is untouched by construction: galls are appended, nothing is moved),
  undercut fraction, and synthetic gall size vs the REAL galls visible in the same photo
  (connected gall-coloured components, extent / L_ear) -- a size plausibility check only.
Reported as constraint compliance + plausibility; no claim that the galls match the specimen.

usage (sam3d env): CUDA_VISIBLE_DEVICES=0 LIDRA_SKIP_INIT=true ~/micromamba/envs/sam3d/bin/python experiments/review/b3_corn_galls.py
  -> results/review/b3_corn/<sid>/<arm>/asset.ply, results/review/b3_corn/summary.{json,md}
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/experiments/demos")
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
from geometry_editor.anatomy import ear_frame  # noqa: E402
from geometry_editor import placement, gall  # noqa: E402
from geometry_editor.evidence_policy import decide  # noqa: E402
from metrics.structure3d import input_view_iou  # noqa: E402

OUT = f"{ROOT}/results/review/b3_corn"
EARS = ["Corn__Smut__0012", "Corn__Smut__0052"]
SEEDS = [0, 1, 2]


def real_galls(E):
    from run_demos import real_gall_components
    L = float(np.ptp(E["a"]))
    return [max(c["extent_cm"]) / L for c in real_gall_components(E, 1.0)]   # cm_per_unit=1 -> asset units


def one(E, sid, arm, n, s, hidden):
    L = float(np.ptp(E["a"])); d = f"{OUT}/{sid}/{arm}_s{s}"; os.makedirs(d, exist_ok=True)
    C, N, Dm, info, glog = placement.gall_centres(E, n=n, rng=np.random.default_rng(s), hidden_only=hidden)
    if len(C) == 0:
        return dict(arm=arm, seed=s, n_requested=n, n_galls=0, status="no hidden kernel tissue")
    rep = gall.build(E, C, N, Dm, f"{d}/asset.ply", rng=np.random.default_rng(s))
    vol = sum(g["volume"] for g in rep["galls"])
    vis = E["visible"][cKDTree(E["X"]).query(C)[1]]
    return dict(arm=arm, seed=s, n_requested=n, n_galls=len(rep["galls"]),
                volume_per_mille_L3=1000 * vol / L ** 3,
                eq_diam_over_L=[g["eq_diameter"] / L for g in rep["galls"]],
                centres_visible_frac=float(vis.mean()),
                kernels_hidden_frac=float(sum(g["kernels_replaced"] for g in rep["galls"]) / max(1, E["kernel"].sum())),
                undercut_frac=float(np.mean([g["undercut_frac"] for g in rep["galls"]])),
                input_view_iou=input_view_iou(f"{d}/asset.ply", sid)[0])


def preserve_checked(E, sid, s, base, tau, n=3, tries=12):
    """Vp: render-checked preserve mode (the L_obs term as a hard constraint). Sample a hidden-kernel
    placement with KB-range sizes, render the input view, keep it only if the fitted-IoU drop <= tau;
    up to `tries` samples, else abstain. Reports what the photo forces (e.g. smaller galls)."""
    L = float(np.ptp(E["a"])); d = f"{OUT}/{sid}/Vp_s{s}"; os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(1000 + s); log = []
    for t in range(tries):
        C, N, Dm, info, glog = placement.gall_centres(E, n=n, rng=rng, hidden_only=True)
        if len(C) == 0:
            break
        rep = gall.build(E, C, N, Dm, f"{d}/asset.ply", rng=np.random.default_rng(s))
        iou = input_view_iou(f"{d}/asset.ply", sid)[0]
        log.append(dict(try_=t, iou=iou, diam_over_L=[float(x) / L for x in Dm]))
        if base - iou <= tau:
            vol = sum(g["volume"] for g in rep["galls"])
            return dict(arm="Vp", seed=s, n_requested=n, n_galls=len(rep["galls"]), tries=t + 1,
                        volume_per_mille_L3=1000 * vol / L ** 3,
                        eq_diam_over_L=[g["eq_diameter"] / L for g in rep["galls"]],
                        undercut_frac=float(np.mean([g["undercut_frac"] for g in rep["galls"]])),
                        input_view_iou=iou, sampled=log)
    os.remove(f"{d}/asset.ply") if os.path.exists(f"{d}/asset.ply") else None
    return dict(arm="Vp", seed=s, n_requested=n, n_galls=0, tries=len(log), status=f"ABSTAIN: no placement within tau after {len(log)} tries",
                sampled=log)


def main():
    os.makedirs(OUT, exist_ok=True)
    tau = json.load(open(f"{ROOT}/results/review/b2_policy/summary.json"))["tau"]
    allres = {}
    for sid in EARS:
        E = ear_frame(sid); base = input_view_iou(E["ply"], sid)[0]; L = float(np.ptp(E["a"]))
        res = dict(specimen=sid, iou_D=base, tau=tau, real_gall_extent_over_L=real_galls(E), rows=[])
        demo = json.load(open(f"{ROOT}/results/demos/corn/{sid}/metrics.json"))["methods"]
        for s in SEEDS:                                  # reuse Test-2 V / Vr (same code path), scale-free
            for arm in ("V", "Vr"):
                m = demo.get(f"{arm}_s{s}")
                if m:
                    res["rows"].append(dict(arm=arm, seed=s, n_requested=3, n_galls=m["n_galls"],
                                            volume_per_mille_L3=1000 * m["total_volume_cm3"] / 18.0 ** 3,
                                            eq_diam_over_L=[x / 18.0 for x in m["eq_diam_cm"]],
                                            input_view_iou=m["input_view_iou"], undercut_frac=m["undercut_frac"],
                                            centres_on_kernel_frac=m["centres_on_kernel_frac"]))
            for n in (1, 3, 6):
                res["rows"].append(one(E, sid, f"Vh_n{n}", n, s, hidden=True))
            res["rows"].append(preserve_checked(E, sid, s, base, tau))
        for r in res["rows"]:
            if "input_view_iou" in r:
                r["policy"] = decide(base, r["input_view_iou"], tau=tau).decision
            print(sid, r["arm"], r["seed"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                             if k in ("n_galls", "volume_per_mille_L3", "input_view_iou", "policy", "centres_visible_frac")}, flush=True)
        allres[sid] = res
        json.dump(res, open(f"{OUT}/{sid}/metrics.json", "w"), indent=1, default=float)
    json.dump(allres, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    write_md(allres)


def write_md(allres):
    L = ["# B3: corn galls -- preserve-mode placement and severity sweep (scale-free)", "",
         "Volume = added gall volume / (ear length)^3 in per mille; IoU = fitted input-view IoU vs the real mask;",
         "policy = observed-evidence decision in preserve mode (accept / refuse at TAU from B2).", "",
         "| ear | arm | seeds | galls | volume (per mille L^3) | gall eq. diam / L | IoU D -> edit | accepted |",
         "|---|---|---|---|---|---|---|---|"]
    for sid, res in allres.items():
        arms = {}
        for r in res["rows"]:
            arms.setdefault(r["arm"], []).append(r)
        for arm, rs in arms.items():
            ok = [r for r in rs if "input_view_iou" in r]
            if not ok:
                L.append(f"| {sid} | {arm} | {len(rs)} | 0 | - | - | - | {rs[0].get('status', '')} |"); continue
            dia = [x for r in ok for x in r["eq_diam_over_L"]]
            L.append(f"| {sid} | {arm} | {len(ok)} | {np.mean([r['n_galls'] for r in ok]):.1f} | "
                     f"{np.median([r['volume_per_mille_L3'] for r in ok]):.1f} | {np.median(dia):.3f} | "
                     f"{res['iou_D']:.2f} -> {np.median([r['input_view_iou'] for r in ok]):.2f} | "
                     f"{sum(r.get('policy') == 'accept' for r in ok)}/{len(ok)} |")
        rg = res["real_gall_extent_over_L"]
        L.append(f"| {sid} | real galls in photo | - | {len(rg)} | - | {('%.3f' % np.median(rg)) if rg else '-'} (extent) | - | - |")
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
