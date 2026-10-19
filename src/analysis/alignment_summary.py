"""Summarise the A0-A3 alignment study. Corrected pose A3* = A3 if IoU(A3) > IoU(A2) else A2 (SAM 3D's own acceptance rule).
Failure label: reconstruction_failure if IoU(A3*) < 0.6. NOTE: this threshold was chosen by the analyst AFTER seeing the
A0-A3 values (it is not pre-registered). Sensitivity: at 0.7, Peach 0063 (0.67) would also be a failure."""
import json, glob, os
import numpy as np
rows = []
for f in sorted(glob.glob("results/alignment/*/alignment.json")):
    d = json.load(open(f)); m = d["metrics"]; s = f.split("/")[-2]
    star = "A3_render_compare" if m["A3_render_compare"]["iou"] > m["A2_closed_form"]["iou"] else "A2_closed_form"
    r = dict(specimen=s, K_reproj_px=d["K_pointmap_reproj_px"], selected=star, A3_diverged=m["A3_render_compare"]["iou"] < m["A2_closed_form"]["iou"] - 0.05)
    for a in ("A0", "A1_pp_centre", "A2_closed_form", "A3_render_compare"):
        for k in ("iou", "bf", "centroid_err", "area_err", "bbox_iou", "lpips", "delta_e", "depth_rel_err"):
            r[f"{a.split('_')[0]}_{k}"] = m[a][k]
    for k in ("iou", "bf", "centroid_err", "area_err", "bbox_iou", "lpips", "delta_e", "depth_rel_err"):
        r[f"A3s_{k}"] = m[star][k]
    r["label"] = "reconstruction_failure" if r["A3s_iou"] < 0.6 else "ok"
    rows.append(r)
json.dump(rows, open("results/alignment/summary.json", "w"), indent=1)
reported = {"Citrus__Canker__0012": .73, "Citrus__Canker__0016": .76, "Corn__Smut__0012": .63, "Corn__Smut__0025": .61, "Corn__Smut__0052": .72,
            "Peach__Leaf_Curl__0006": .43, "Peach__Leaf_Curl__0063": .58, "Peach__Leaf_Curl__0087": .84, "Tomato__Leaf_Mold__0015": .40,
            "Tomato__Leaf_Mold__0052": .55, "Tomato__Leaf_Mold__0092": .47}
print("| Specimen | reported | A0 IoU | A1 IoU | A2 IoU | A3 IoU | **A3\\* IoU** | BF A0→A3* | area err A0 | LPIPS A0→A3* (held-out) | depth err A0→A3* (held-out) | label |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    rep = reported.get(r["specimen"]); 
    print(f"| {r['specimen'].replace('__', ' ')} | {rep if rep else '—'} | {r['A0_iou']:.2f} | {r['A1_iou']:.2f} | {r['A2_iou']:.2f} | {r['A3_iou']:.2f}{' (diverged)' if r['A3_diverged'] else ''} | **{r['A3s_iou']:.2f}** | {r['A0_bf']:.2f}→{r['A3s_bf']:.2f} | {r['A0_area_err']:+.2f} | {r['A0_lpips']:.2f}→{r['A3s_lpips']:.2f} | {r['A0_depth_rel_err']:.3f}→{r['A3s_depth_rel_err']:.3f} | {r['label']} |")
a0 = np.array([r["A0_iou"] for r in rows]); a3 = np.array([r["A3s_iou"] for r in rows])
print(f"\nMedian IoU A0 {np.median(a0):.2f} -> A3* {np.median(a3):.2f}; improved {int((a3 > a0 + 0.01).sum())}/{len(rows)}, degraded {int((a3 < a0 - 0.01).sum())}/{len(rows)}; "
      f"held-out depth error worse in {sum(r['A3s_depth_rel_err'] > r['A0_depth_rel_err'] + 0.01 for r in rows)}/{len(rows)}, LPIPS worse (>0.02) in {sum(r['A3s_lpips'] > r['A0_lpips'] + 0.02 for r in rows)}/{len(rows)}")
