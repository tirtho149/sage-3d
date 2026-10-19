"""Review item B1: directional curl, evaluated independently of the operator target, on every leaf.

The Test-1 demo checked the curl operator on 3 Mango leaves with one summary number (median
margin angle), which mostly verifies that the operator applies the angle it is given. Here:
  * all 12 pilot leaves (Peach, Tomato, Citrus, Mango; Peach 0006 is the known ring failure,
    kept and flagged), cited direction (CU) and reversed control (CD), seed 0;
  * the roll PROFILE along the leaf (10 inner u-bins), not one median: fraction of positions
    whose roll moved in the commanded direction, and its spread;
  * tissue strain and fine-relief preservation (structure3d.deformation_report);
  * self-contact: points that touch in 3D (< 1% L apart) although they are far apart on
    the sheet (> 8% L in the original flattened (u, lateral) coordinates), relative to the
    unedited asset, so reconstruction folds are not blamed on the edit;
  * midrib preservation: arc-length ratio and shape RMS (after rigid alignment) of the
    midrib polyline (|r| < 0.1, 24 u-bins);
  * fitted input-view IoU vs the real mask (how much the edit contradicts the photo).
The angle itself (theta from the KB phrase parser) is NOT used as evidence: it is imposed.

usage (sam3d env, from the repo root):
  CUDA_VISIBLE_DEVICES=0 LIDRA_SKIP_INIT=true ~/micromamba/envs/sam3d/bin/python experiments/review/b1_curl_robustness.py
  -> results/review/b1_curl/<sid>/{CU,CD}/asset.ply, results/review/b1_curl/summary.{json,md}
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src")
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
from geometry_editor.anatomy import leaf_frame, load_vertices, xyz  # noqa: E402
from geometry_editor.curl import parse, run_curl  # noqa: E402
from metrics.structure3d import roll_metrics, deformation_report, input_view_iou, _kabsch  # noqa: E402

OUT = f"{ROOT}/results/review/b1_curl"
PHRASE = "Leaves curl upwards and fall off."          # same SAGE phrase as Test 1 (Mango Die Back)
LEAVES = [f"{c}__{i}" for c, ids in [("Peach__Leaf_Curl", ["0006", "0063", "0087"]),
                                      ("Tomato__Leaf_Mold", ["0015", "0052", "0092"]),
                                      ("Citrus__Canker", ["0004", "0012", "0016"]),
                                      ("Mango__Anthracnose", ["0044", "0049", "0094"])] for i in ids]
KNOWN_FAILURES = {"Peach__Leaf_Curl__0006": "SAM 3D ring topology (E1)"}


def self_contact(F, Y, n=8000, d3=0.01, dp=0.08, seed=0):
    """Fraction of sampled points with a partner < d3*L away in 3D but > dp*L apart on the sheet."""
    L = F["L"]; P = np.stack([F["u"], F["lat"]], 1)
    idx = np.random.default_rng(seed).choice(len(Y), min(n, len(Y)), replace=False)
    tree = cKDTree(Y[idx]); pairs = tree.query_pairs(d3 * L, output_type="ndarray")
    if len(pairs) == 0:
        return 0.0
    far = np.linalg.norm(P[idx[pairs[:, 0]]] - P[idx[pairs[:, 1]]], axis=1) > dp * L
    hit = np.zeros(len(idx), bool); hit[pairs[far, 0]] = True; hit[pairs[far, 1]] = True
    return float(hit.mean())


def midrib(F, Y, bins=24):
    X, u, r, L = F["X"], F["u"], F["r"], F["L"]
    mid = np.abs(r) < 0.1
    edges = np.quantile(u[mid], np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(edges, u[mid], side="right") - 1, 0, bins - 1)
    keep = [i for i in range(bins) if (b == i).sum() >= 5]
    A = np.array([X[mid][b == i].mean(0) for i in keep]); B = np.array([Y[mid][b == i].mean(0) for i in keep])
    la = np.linalg.norm(np.diff(A, axis=0), axis=1).sum(); lb = np.linalg.norm(np.diff(B, axis=0), axis=1).sum()
    Rm = _kabsch(B - B.mean(0), A - A.mean(0))
    rms = np.sqrt((((B - B.mean(0)) @ Rm.T - (A - A.mean(0))) ** 2).sum(1).mean()) / L
    return dict(midrib_length_ratio=float(lb / max(la, 1e-12)), midrib_shape_rms_L=float(rms))


def evaluate(F, Y, base_roll):
    m = roll_metrics(F, Y)
    m.update(deformation_report(F, Y))
    m.update(midrib(F, Y))
    m["self_contact_frac"] = self_contact(F, Y)
    if base_roll is not None:
        b = {p["bin"]: p["mean_deg"] for p in base_roll}
        d = [p["mean_deg"] - b[p["bin"]] for p in m["roll_profile"] if p["bin"] in b]
        m["roll_change_profile"] = d
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    P = parse(PHRASE)
    rows = {}
    for sid in LEAVES:
        res = dict(specimen=sid, phrase=PHRASE, parsed=P, flag=KNOWN_FAILURES.get(sid))
        try:
            F = leaf_frame(sid)
        except Exception as exc:  # noqa: BLE001
            res["error"] = f"leaf_frame failed: {exc}"; rows[sid] = res; print(sid, res["error"]); continue
        base = evaluate(F, F["X"], None)
        base["input_view_iou"] = input_view_iou(F["ply"], sid)[0]
        res["D"] = base
        for name, sg in (("CU", P["direction"]), ("CD", -P["direction"])):
            d = f"{OUT}/{sid}/{name}"
            if not os.path.exists(f"{d}/asset.ply"):
                run_curl(F, sg, P["theta_deg"], P["r0"], d, f"{name}:{PHRASE}", seed=0, iters=1000)
            Y = xyz(load_vertices(f"{d}/asset.ply"))
            m = evaluate(F, Y, base["roll_profile"])
            m["input_view_iou"] = input_view_iou(f"{d}/asset.ply", sid)[0]
            ch = np.array(m.get("roll_change_profile", []))
            m["frac_positions_commanded_sign"] = float((np.sign(ch) == sg).mean()) if len(ch) else None
            m["median_roll_change"] = float(np.median(ch)) if len(ch) else None
            m["self_contact_added"] = m["self_contact_frac"] - base["self_contact_frac"]
            res[name] = m
            print(sid, name, {k: round(m[k], 3) for k in ("median_roll_change", "frac_positions_commanded_sign",
                                                         "strain_02L_p95", "self_contact_added",
                                                         "midrib_length_ratio", "input_view_iou")
                               if m.get(k) is not None}, flush=True)
        rows[sid] = res
        json.dump(res, open(f"{OUT}/{sid}/metrics.json", "w"), indent=1, default=float)
    json.dump(rows, open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    write_md(rows)


def write_md(rows):
    L = ["# B1: directional curl on all 12 leaves (independent measures)", "",
         f"Phrase: \"{PHRASE}\"; CU = cited direction, CD = reversed control; seed 0. The imposed angle is not evidence.", "",
         "| specimen | arm | median roll change (deg) | positions in commanded direction | strain p95 | fine-relief corr | self-contact added | midrib length ratio | midrib shape RMS/L | input-view IoU (D -> edit) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for sid, r in rows.items():
        if "error" in r:
            L.append(f"| {sid} | - | {r['error']} | | | | | | | |"); continue
        for arm in ("CU", "CD"):
            m = r[arm]; f = lambda k, p=2: (f"{m[k]:.{p}f}" if m.get(k) is not None else "-")
            L.append(f"| {sid}{' (' + r['flag'] + ')' if r.get('flag') else ''} | {arm} | {f('median_roll_change', 1)} | "
                     f"{f('frac_positions_commanded_sign')} | {f('strain_02L_p95', 3)} | {f('fine_relief_corr')} | "
                     f"{f('self_contact_added', 3)} | {f('midrib_length_ratio', 3)} | {f('midrib_shape_rms_L', 3)} | "
                     f"{r['D']['input_view_iou']:.2f} -> {f('input_view_iou')} |")
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
