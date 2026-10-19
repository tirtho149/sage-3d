"""Slide macros for the review-response results (results/review/*/summary.json).

usage: python3 experiments/review/review_macros.py  -> figures/slides/review_numbers.tex
Missing summaries are skipped (their macros are not defined, so the deck fails loudly).
"""
import json
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RV = f"{ROOT}/results/review"


def load(p):
    p = f"{RV}/{p}"
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    M = {}
    b2 = load("b2_policy/summary.json")
    if b2:
        M["PolTau"] = f"{b2['tau']:.3f}"
        e = b2["by_edit"]
        def cnt(key, field="accept"):
            v = e[key]; return f"{v[field]}/{v['n']}"
        M["PolPucker"] = f"{e['peach|KB pucker, evidence-placed']['accept'] + e['peach|KB pucker, uniform']['accept']}/" \
                         f"{e['peach|KB pucker, evidence-placed']['n'] + e['peach|KB pucker, uniform']['n']}"
        M["PolTomato"] = f"{sum(e[k]['accept'] for k in e if k.startswith('tomato|'))}/{sum(e[k]['n'] for k in e if k.startswith('tomato|'))}"
        M["PolMangoRefused"] = f"{e['mango_curl|KB curl (cited dir.)']['refuse'] + e['mango_curl|reversed curl (control)']['refuse']}/" \
                               f"{e['mango_curl|KB curl (cited dir.)']['n'] + e['mango_curl|reversed curl (control)']['n']}"
        M["PolCornRefused"] = cnt("corn|volumetric galls on kernels", "refuse")
        M["PolMangoDrop"] = f"{st.median(e['mango_curl|KB curl (cited dir.)']['drops']):.2f}"
    b3 = load("b3_corn/summary.json")
    if b3:
        r12 = b3["Corn__Smut__0012"]["rows"]
        vp = [r for sid in b3 for r in b3[sid]["rows"] if r["arm"] == "Vp" and r.get("n_galls")]
        vh = [r for r in r12 if r["arm"] == "Vh_n3"]
        M["GallVpAccepted"] = f"{len(vp)}/{sum(1 for sid in b3 for r in b3[sid]['rows'] if r['arm'] == 'Vp')}"
        M["GallVpTries"] = f"{min(r['tries'] for r in vp)}--{max(r['tries'] for r in vp)}"
        d12_vp = [x for r in r12 if r["arm"] == "Vp" for x in r.get("eq_diam_over_L", [])]
        d12_v = [x for r in r12 if r["arm"] == "V" for x in r.get("eq_diam_over_L", [])]
        M["GallDiamVp"] = f"{st.median(d12_vp):.3f}"; M["GallDiamV"] = f"{st.median(d12_v):.3f}"
        M["GallVolVp"] = f"{st.median([r['volume_per_mille_L3'] for r in r12 if r['arm'] == 'Vp']):.1f}"
        M["GallVolV"] = f"{st.median([r['volume_per_mille_L3'] for r in r12 if r['arm'] == 'V']):.1f}"
        M["GallIouD"] = f"{b3['Corn__Smut__0012']['iou_D']:.2f}"
        M["GallIouVh"] = f"{st.median([r['input_view_iou'] for r in vh]):.2f}"
        # per-ear median extent of the real (photo) galls; single components can merge several galls or husk
        real = [st.median(b3[sid]["real_gall_extent_over_L"]) for sid in b3 if b3[sid]["real_gall_extent_over_L"]]
        M["GallReal"] = f"{min(real):.2f}--{max(real):.2f}" if real else "--"
    b5 = load("b5_statistics/summary.json")
    if b5:
        c = {f"{r['comparison']}|{r['metric']}": r for r in b5["comparisons"]}
        g = c["J vs G|geometry_recall"]
        M["StFloor"] = f"{g['exact_p_floor']:.3f}"; M["StHolmFloor"] = f"{g['holm_p_floor']:.3f}"
        M["StNonzero"] = str(g["n_nonzero"]); M["StN"] = str(g["n"]); M["StTests"] = str(b5["n_tests_holm"])
    b6 = load("b6_shared_texture/summary.json")
    if b6:
        ok = [v for v in b6.values() if "crossview_Ctex" in v]
        M["CtexCv"] = f"{st.median(v['crossview_Ctex'] for v in ok):.1f}"
        M["CtexdE"] = f"{st.median(v['edit_dE_obj'] for v in ok):.0f}"
        M["CtexN"] = str(len(ok))
    b7 = load("b7_warp2d/summary.json")
    if b7:
        v = list(b7.values())
        M["WarpTwoD"] = f"{min(x['dE_updown_2D'] for x in v):.0f}--{max(x['dE_updown_2D'] for x in v):.0f}"
        M["WarpThreeD"] = f"{min(x['dE_updown_3D_rig'] for x in v):.0f}--{max(x['dE_updown_3D_rig'] for x in v):.0f}"
        M["WarpIou"] = f"{min(min(x['iou_2D_up'], x['iou_2D_down']) for x in v):.2f}--{max(max(x['iou_2D_up'], x['iou_2D_down']) for x in v):.2f}"
    b8 = load("b8_domain_gap/summary.json")
    if b8:
        s = list(b8["steps"].values())
        M["GapStepRender"] = f"{s[0]:+.2f}"; M["GapStepBg"] = f"{s[1]:+.2f}"; M["GapStepView"] = f"{s[2]:+.2f}"; M["GapStepTex"] = f"{s[3]:+.2f}"
    b9 = load("b9_ablation/summary.json")
    if b9:
        mg, pc = b9["Mango__Anthracnose__0049"], b9["Peach__Leaf_Curl__0087"]
        M["AblStrainFull"] = f"{mg['full']['strain_02L_p95']:.2f}/{pc['full']['strain_02L_p95']:.2f}"
        M["AblStrainNoArap"] = f"{mg['no_arap']['strain_02L_p95']:.2f}/{pc['no_arap']['strain_02L_p95']:.2f}"
    b4 = load("b4_surfaces/summary.json")
    if b4:
        L = b4["leaves"]
        tom = [v for k, v in L.items() if k.startswith("Tomato")]
        M["SurfTomMixed"] = f"{min(v['mixed_views'] for v in tom)}--{max(v['mixed_views'] for v in tom)}"
        M["SurfAllMixed"] = f"{min(v['mixed_views'] for v in L.values())}--{max(v['mixed_views'] for v in L.values())}"
        M["SurfTomShare"] = f"{100 * st.median(v['mean_minority_frac'] for v in tom):.0f}"
        M["SurfShareRange"] = f"{100 * min(v['mean_minority_frac'] for v in L.values()):.0f}--{100 * max(v['mean_minority_frac'] for v in L.values()):.0f}"
        M["SurfTomRange"] = f"{100 * min(v['mean_minority_frac'] for v in tom):.0f}--{100 * max(v['mean_minority_frac'] for v in tom):.0f}"
    b1 = load("b1_curl/summary.json")
    if b1:
        ok = [r for sid, r in b1.items() if "CU" in r and not r.get("flag")]
        M["CurlN"] = str(len(ok))
        M["CurlDirOK"] = str(sum(1 for r in ok if r["CU"]["median_roll_change"] > 0 and r["CD"]["median_roll_change"] < 0))
        M["CurlPosFrac"] = f"{100 * st.median([r['CU']['frac_positions_commanded_sign'] for r in ok] + [r['CD']['frac_positions_commanded_sign'] for r in ok]):.0f}"
        M["CurlStrain"] = f"{st.median(r['CU']['strain_02L_p95'] for r in ok):.2f}"
        M["CurlContact"] = f"{max(max(r['CU']['self_contact_added'], r['CD']['self_contact_added']) for r in ok):.3f}"
        M["CurlMidrib"] = f"{min(min(r['CU']['midrib_length_ratio'], r['CD']['midrib_length_ratio']) for r in ok):.3f}"
        M["CurlIouAccepted"] = str(sum(1 for r in ok for a in ("CU", "CD") if r["D"]["input_view_iou"] - r[a]["input_view_iou"] <= (b2 or {}).get("tau", 0.046)))
        M["CurlIouTotal"] = str(2 * len(ok))
        mags = [abs(r[a]["median_roll_change"]) for r in ok for a in ("CU", "CD")]
        M["CurlMag"] = f"{min(mags):.0f}--{max(mags):.0f}"
        bad = [sid for sid, r in b1.items() if "CU" in r and not r.get("flag") and max(r["CU"]["strain_02L_p95"], r["CD"]["strain_02L_p95"]) > 1.0]
        M["CurlBadN"] = str(len(bad))
        M["CurlBadStrain"] = f"{min(min(b1[s_]['CU']['strain_02L_p95'], b1[s_]['CD']['strain_02L_p95']) for s_ in bad):.1f}--{max(max(b1[s_]['CU']['strain_02L_p95'], b1[s_]['CD']['strain_02L_p95']) for s_ in bad):.1f}" if bad else "--"
        good = [r for sid, r in b1.items() if "CU" in r and not r.get("flag") and sid not in bad]
        M["CurlGoodStrain"] = f"{min(min(r['CU']['strain_02L_p95'], r['CD']['strain_02L_p95']) for r in good):.2f}--{max(max(r['CU']['strain_02L_p95'], r['CD']['strain_02L_p95']) for r in good):.2f}"
        drops = [r["D"]["input_view_iou"] - r[a]["input_view_iou"] for r in ok for a in ("CU", "CD")]
        M["CurlIouDrop"] = f"{min(drops):.2f}--{max(drops):.2f}"
    tex = ["% generated by experiments/review/review_macros.py from results/review/*/summary.json -- do not edit by hand"]
    tex += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    open(f"{ROOT}/figures/slides/review_numbers.tex", "w").write("\n".join(tex) + "\n")
    print(f"{len(M)} review macros")
    for k, v in M.items():
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
