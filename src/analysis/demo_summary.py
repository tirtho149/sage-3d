"""Aggregate results/demos/*/<specimen>/metrics.json into results/demos/summary.json (+ summary.md table).

Unit = specimen (seeds averaged within specimen first). With 2-3 specimens per demo the summaries are descriptive:
we report specimen means, the per-specimen values, and how many specimens show the expected sign; no p-values.
"""
import os, json, glob
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); D = f"{R}/results/demos"


def seedmean(methods, prefix, key):
    v = [m[key] for k, m in methods.items() if (k == prefix or k.startswith(prefix + "_s")) and m.get(key) is not None and not isinstance(m.get(key), (list, dict))]
    return float(np.mean(v)) if v else None


def per_spec(demo, specs, keys):
    out = {}
    for p in sorted(glob.glob(f"{D}/{demo}/*/metrics.json")):
        d = json.load(open(p)); sid = d["specimen"]; out[sid] = {}
        for meth in specs:
            out[sid][meth] = {k: seedmean(d["methods"], meth, k) for k in keys}
        out[sid]["_status"] = d.get("status", "ok")
    return out


def agg(ps, meth, key):
    v = [ps[s][meth][key] for s in ps if meth in ps[s] and ps[s][meth].get(key) is not None]
    return dict(mean=float(np.mean(v)) if v else None, values=[round(x, 4) for x in v], n=len(v))


S = {}
# tomato: surface placement
tk = ["mold_on_upper", "mold_on_lower", "spot_on_lower", "spot_on_upper", "input_view_iou", "mold_footprint_on_spots"]
tm = ["REAL_photo", "B_2D_generic", "C_2D_kb_joint", "Cstage_2D_kb", "D", "E", "F", "K", "BL0", "BLk", "BLr", "BLu"]
ps = per_spec("tomato", tm, tk); S["tomato"] = {m: {k: agg(ps, m, k) for k in tk} for m in tm}; S["tomato_per_specimen"] = ps
S["tomato"]["_n_specimens"] = len(ps)
# mango: direction control
mk = ["roll_deg_median", "kappa_t_halfwidth_median", "strain_02L_p95", "fine_relief_corr", "input_view_iou"]
ps = per_spec("mango_curl", ["D", "G", "CU_kb", "CD_opposite"], mk); S["mango"] = {m: {k: agg(ps, m, k) for k in mk} for m in ["D", "G", "CU_kb", "CD_opposite"]}
S["mango"]["_direction_agreement"] = dict(
    kb=sum(1 for s in ps if ps[s]["CU_kb"]["roll_deg_median"] is not None and ps[s]["CU_kb"]["roll_deg_median"] > ps[s]["D"]["roll_deg_median"]),
    opposite_control_reversed=sum(1 for s in ps if ps[s]["CD_opposite"]["roll_deg_median"] is not None and ps[s]["CD_opposite"]["roll_deg_median"] < ps[s]["D"]["roll_deg_median"]), n=len(ps))
S["mango_per_specimen"] = ps
# peach
pk = ["roll_deg_median", "kappa_t_halfwidth_median", "strain_02L_p95", "fine_relief_corr", "input_view_iou", "placement_enrichment", "pucker_energy_on_reddish", "reddish_area_frac", "disp_rms_rel"]
pm = ["D", "J", "G", "E", "F", "K", "H", "J2", "J2u", "J2+curl_up_HYP", "J2+curl_down_HYP"]
ps = per_spec("peach", pm, pk); S["peach"] = {m: {k: agg(ps, m, k) for k in pk} for m in pm}; S["peach_per_specimen"] = ps
# corn
ck = ["input_view_iou", "centres_on_kernel_frac", "total_volume_cm3", "undercut_frac", "kernels_replaced", "added_volume_heightfield", "max_offset_rel"]
cm_ = ["D", "J", "G", "E", "F", "V", "Vr"]
ps = per_spec("corn", cm_, ck); S["corn"] = {m: {k: agg(ps, m, k) for k in ck} for m in cm_}; S["corn_per_specimen"] = ps
corn_scale, real_galls = {}, {}
for p in glob.glob(f"{D}/corn/*/metrics.json"):
    d = json.load(open(p)); corn_scale[d["specimen"]] = d.get("scale"); real_galls[d["specimen"]] = d.get("real_galls_in_photo"); S.setdefault("corn_status", {})[d["specimen"]] = d.get("status", "ok")
    # height-field volume in cm^3 under the same scale assumption
    if d.get("scale") and "J" in d["methods"] and d["methods"]["J"].get("added_volume_heightfield") is not None:
        S.setdefault("corn_J_heightfield_cm3", {})[d["specimen"]] = d["methods"]["J"]["added_volume_heightfield"] * d["scale"]["cm_per_unit"] ** 3
S["corn_scale"] = corn_scale; S["corn_real_galls_in_photo"] = real_galls
json.dump(S, open(f"{D}/summary.json", "w"), indent=1, default=float)


def f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


L = ["# Demo summary (specimen = unit; seeds averaged within specimen)", ""]
L += ["## Tomato Leaf Mold: where the mold-coloured pixels land", "", "| method | mold on UPPER | mold on LOWER | upper spots on LOWER | footprint on spots | n |", "|---|---|---|---|---|---|"]
for m in tm:
    a = S["tomato"][m]; L.append(f"| {m} | {f(a['mold_on_upper']['mean'])} | {f(a['mold_on_lower']['mean'])} | {f(a['spot_on_lower']['mean'])} | {f(a['mold_footprint_on_spots']['mean'])} | {a['mold_on_upper']['n']} |")
L += ["", "## Mango leaves + 'Leaves curl upwards' (direction control)", "", "| method | roll (deg) | kappa_t*w | strain p95 | relief corr | input IoU | n |", "|---|---|---|---|---|---|---|"]
for m in ["D", "G", "CU_kb", "CD_opposite"]:
    a = S["mango"][m]; L.append(f"| {m} | {f(a['roll_deg_median']['mean'], 1)} | {f(a['kappa_t_halfwidth_median']['mean'], 2)} | {f(a['strain_02L_p95']['mean'])} | {f(a['fine_relief_corr']['mean'])} | {f(a['input_view_iou']['mean'])} | {a['roll_deg_median']['n']} |")
L += ["", f"direction: KB-direction edits raised the margins in {S['mango']['_direction_agreement']['kb']}/{S['mango']['_direction_agreement']['n']} specimens; the opposite-direction control lowered them in {S['mango']['_direction_agreement']['opposite_control_reversed']}/{S['mango']['_direction_agreement']['n']}.", ""]
L += ["## Peach Leaf Curl", "", "| method | roll (deg) | strain p95 | relief corr | input IoU | pucker enrichment on reddish tissue | n |", "|---|---|---|---|---|---|---|"]
for m in pm:
    a = S["peach"][m]; L.append(f"| {m} | {f(a['roll_deg_median']['mean'], 1)} | {f(a['strain_02L_p95']['mean'])} | {f(a['fine_relief_corr']['mean'])} | {f(a['input_view_iou']['mean'])} | {f(a['placement_enrichment']['mean'], 2)} | {a['input_view_iou']['n']} |")
L += ["", "## Corn Smut", "", "| method | galls on kernel tissue | total gall volume (cm^3, assumed scale) | undercut fraction | input IoU | n |", "|---|---|---|---|---|---|"]
for m in cm_:
    a = S["corn"][m]; L.append(f"| {m} | {f(a['centres_on_kernel_frac']['mean'], 2)} | {f(a['total_volume_cm3']['mean'], 1)} | {f(a['undercut_frac']['mean'], 2)} | {f(a['input_view_iou']['mean'])} | {a['input_view_iou']['n']} |")
L += ["", f"height-field J added volume (cm^3, same scale): {json.dumps({k: round(v, 2) for k, v in S.get('corn_J_heightfield_cm3', {}).items()})}",
      f"status: {json.dumps(S.get('corn_status', {}))}"]
open(f"{D}/summary.md", "w").write("\n".join(L) + "\n"); print("\n".join(L))
