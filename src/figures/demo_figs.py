"""Three advisor figures from results/demos (brief v3 item 10). Drawn at slide size (143 mm wide) so text is 6-7 pt
on the slide; vector PDF + 300 dpi PNG to figures/slides/. Every image panel carries a provenance tag:
REAL (the SAGE photo), EDITED REAL (2D edit of the photo), SYNTHETIC (render of a 3D asset; unverified).
usage (sam3d env): python src/figures/demo_figs.py
"""
import os, sys, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, f"{R}/src")
OUT = f"{R}/figures/slides"; os.makedirs(OUT, exist_ok=True); D = f"{R}/results/demos"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.3, "xtick.labelsize": 5.8,
                     "ytick.labelsize": 5.8, "legend.fontsize": 5.8, "pdf.fonttype": 42, "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False})
CARD, GOLD, GRAY, GREEN, BROWN, BLUE = "#C8102E", "#F1BE48", "#707372", "#2E7D32", "#8B5B29", "#006BA6"
TAG = {"REAL": GREEN, "EDITED REAL (2D)": BROWN, "SYNTHETIC": CARD, "HYPOTHESIS": "#BE531C"}
S = json.load(open(f"{D}/summary.json"))


def crop(a, pad=6, thr=246):
    m = (a < thr).any(-1); ys, xs = np.nonzero(m)
    if len(ys) == 0: return a
    y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + pad, max(0, xs.min() - pad), xs.max() + pad
    h, w = y1 - y0, x1 - x0; s = max(h, w); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    out = np.full((s, s, 3), 255, np.uint8); ys0, xs0 = max(0, cy - s // 2), max(0, cx - s // 2)
    sub = a[ys0:ys0 + s, xs0:xs0 + s]; out[:sub.shape[0], :sub.shape[1]] = sub; return out


def im(p, white_crop=True):
    if not os.path.exists(p): return None
    a = np.asarray(Image.open(p).convert("RGB")); return crop(a) if white_crop else a


def panel(ax, a, title, tag):
    ax.axis("off")
    if a is None: ax.text(.5, .5, "n/a", ha="center", va="center", color=GRAY, transform=ax.transAxes); return
    ax.imshow(a, interpolation="lanczos"); ax.set_title(title, pad=1.5, fontsize=6.2)
    ax.text(0.02, 0.02, tag, transform=ax.transAxes, fontsize=4.8, color="white", fontweight="bold", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc=TAG[tag], ec="none"))


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)


# ------------------------------------------------------------------------------- (a) directional geometry
def cross_section(sid, plies, frac=0.12):
    """Midrib cross-section (points in the middle u-band): lateral vs adaxial-normal coordinate, original frame."""
    from geometry_editor.anatomy import leaf_frame, xyz, load_vertices
    F = leaf_frame(sid); u = F["u"]; band = np.abs(u - np.median(u)) < frac * np.ptp(u) / 2
    mid = band & (np.abs(F["r"]) < 0.15); c = F["X"][mid].mean(0); n = F["n"][mid].mean(0); n /= np.linalg.norm(n)
    tp = F["X"][band & (F["r"] > 0.3)].mean(0) - F["X"][band & (F["r"] < -0.3)].mean(0); tp -= (tp @ n) * n; tp /= np.linalg.norm(tp)
    out = {}
    for k, p in plies.items():
        Y = xyz(load_vertices(p))[:len(F["X"])]; P = Y[band] - Y[mid].mean(0)
        sub = np.random.default_rng(0).choice(len(P), min(2500, len(P)), replace=False); out[k] = (P[sub] @ tp / F["L"], P[sub] @ n / F["L"])
    return out


def fig_a():
    msid, psid = "Mango__Anthracnose__0049", "Peach__Leaf_Curl__0063"; md, pd = f"{D}/mango_curl/{msid}", f"{D}/peach/{psid}"
    fig = plt.figure(figsize=(5.65, 2.6)); gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.3, 1.45], wspace=0.06, hspace=0.62)
    row = [(im(f"{R}/data/inputs/{msid}/image.png", False), "Mango photo", "REAL"), (im(f"{md}/D/views/002.png"), "D unedited", "SYNTHETIC"),
           (im(f"{md}/CU_kb_s0/views/002.png"), "KB: curl upwards", "SYNTHETIC"), (im(f"{md}/CD_opposite_s0/views/002.png"), "ctrl: opposite", "SYNTHETIC")]
    for j, (a, t, g) in enumerate(row): panel(fig.add_subplot(gs[0, j]), a, t, g)
    row = [(im(f"{R}/data/inputs/{psid}/image.png", False), "Peach photo", "REAL"), (im(f"{pd}/J2_s0/views/002.png"), "KB v2 pucker", "SYNTHETIC"),
           (im(f"{pd}/J2+curl_up_HYP/views/002.png"), "+ curl up?", "HYPOTHESIS"), (im(f"{pd}/J2+curl_down_HYP/views/002.png"), "+ curl down?", "HYPOTHESIS")]
    for j, (a, t, g) in enumerate(row): panel(fig.add_subplot(gs[1, j]), a, t, g)
    for i, (sid, plies, cols, title) in enumerate([
            (msid, {"D": f"{md}/D/../../../../pilot/sam3d/{msid}/seed0/splat_ge.ply", "KB up": f"{md}/CU_kb_s0/asset.ply", "opposite": f"{md}/CD_opposite_s0/asset.ply"},
             {"D": GRAY, "KB up": CARD, "opposite": BLUE}, "mid-blade section"),
            (psid, {"D": f"{R}/results/pilot/sam3d/{psid}/seed0/splat_ge.ply", "up?": f"{pd}/J2+curl_up_HYP/asset.ply", "down?": f"{pd}/J2+curl_down_HYP/asset.ply"},
             {"D": GRAY, "up?": "#BE531C", "down?": BLUE}, "section (hypotheses)")]):
        ax = fig.add_subplot(gs[i, 5]); cs = cross_section(sid, {k: os.path.normpath(v) for k, v in plies.items()})
        for k, (x, y) in cs.items(): ax.scatter(x, y, s=0.4, color=cols[k], label=k, rasterized=True, linewidths=0)
        ax.set_aspect("equal", adjustable="datalim"); ax.set_title(title, pad=1.5); ax.set_xlabel("lateral / L", labelpad=1); ax.set_ylabel("adaxial / L", labelpad=1)
        ax.legend(frameon=False, markerscale=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, handletextpad=0.1, columnspacing=0.6, borderaxespad=0)
        ax.axhline(0, color=GRAY, lw=0.3, ls=":")
    save(fig, "demo_a_directional")


# ------------------------------------------------------------------------------- (b) volumetric galls
def fig_b():
    sid = "Corn__Smut__0052"; cd = f"{D}/corn/{sid}"; v = "002"
    fig = plt.figure(figsize=(5.65, 1.75)); gs = fig.add_gridspec(1, 7, width_ratios=[1.15, 1, 1, 1, 1, 0.1, 1.15], wspace=0.08)
    # all 3D panels at the corrected INPUT camera (same viewpoint as the photo); seed 1 puts a gall on the camera side
    row = [(im(f"{R}/data/inputs/{sid}/image.png", False), "photo", "REAL"), (im(f"{cd}/J/input_view.png"), "v1 height field", "SYNTHETIC"),
           (im(f"{cd}/F/input_view.png"), "GaussianEditor", "SYNTHETIC"), (im(f"{cd}/V_s1/input_view.png"), "v2 volumetric", "SYNTHETIC"),
           (im(f"{cd}/Vr_s1/input_view.png"), "random site", "SYNTHETIC")]
    for j, (a, t, g) in enumerate(row): panel(fig.add_subplot(gs[0, j]), a, t, g)
    ax = fig.add_subplot(gs[0, 6]); C = S["corn"]
    und = [0.0, C["V"]["undercut_frac"]["mean"] or 0, C["Vr"]["undercut_frac"]["mean"] or 0]
    onk = [0.0, C["V"]["centres_on_kernel_frac"]["mean"] or 0, C["Vr"]["centres_on_kernel_frac"]["mean"] or 0]
    x = np.arange(3); ax.bar(x - 0.2, und, 0.38, color=GRAY, label="overhang"); ax.bar(x[1:] + 0.2, onk[1:], 0.38, color=CARD, label="on kernels")
    ax.set_xticks(x); ax.set_xticklabels(["height\nfield", "v2", "random"]); ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1])
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=0.8, columnspacing=0.6, borderaxespad=0.2)
    ax.text(0.5, -0.42, f"{C['V']['undercut_frac']['n']} ears x 3 seeds", transform=ax.transAxes, ha="center", fontsize=5.4, color=GRAY)
    save(fig, "demo_b_galls")


# ------------------------------------------------------------------------------- (c) underside symptoms
def fig_c():
    sid = "Tomato__Leaf_Mold__0015"; td = f"{D}/tomato/{sid}"
    fig = plt.figure(figsize=(5.65, 2.45)); gs = fig.add_gridspec(2, 6, width_ratios=[1, 1, 1, 1, 0.55, 1.75], wspace=0.06, hspace=0.28)
    top = [(im(f"{R}/data/inputs/{sid}/image.png", False), "photo (upper)", "REAL"), (im(f"{R}/results/phaseA/{sid}/C_stage/seed0.png", False), "2D KB edit", "EDITED REAL (2D)"),
           (im(f"{td}/D/views/000.png"), "D upper", "SYNTHETIC"), (im(f"{td}/BLk_s0/views/000.png"), "2-surface upper", "SYNTHETIC")]
    bot = [(im(f"{td}/D/views/004.png"), "D lower", "SYNTHETIC"), (im(f"{td}/F/views/004.png"), "GE+KB lower", "SYNTHETIC"),
           (im(f"{td}/BL0_s0/views/004.png"), "2-surface lower", "SYNTHETIC"), (im(f"{td}/BLk_s0/views/004.png"), "2-surf. KB lower", "SYNTHETIC")]
    for j, (a, t, g) in enumerate(top): panel(fig.add_subplot(gs[0, j]), a, t, g)
    for j, (a, t, g) in enumerate(bot): panel(fig.add_subplot(gs[1, j]), a, t, g)
    ax = fig.add_subplot(gs[:, 5]); T = S["tomato"]
    meths = [("REAL_photo", "real photo"), ("Cstage_2D_kb", "2D KB C-stage"), ("C_2D_kb_joint", "2D KB C"), ("D", "3D unedited"), ("F", "3D GE+KB"),
             ("BLk", "2-surface KB"), ("BLu", "ctrl: upper")]
    y = np.arange(len(meths))[::-1]
    up = [T[m]["mold_on_upper"]["mean"] or 0 for m, _ in meths]; lo = [T[m]["mold_on_lower"]["mean"] if T[m]["mold_on_lower"]["mean"] is not None else 0 for m, _ in meths]
    ax.barh(y + 0.19, up, 0.36, color=CARD, label="upper (KB: wrong)"); ax.barh(y - 0.19, lo, 0.36, color=GREEN, label="lower (KB: right)")
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in meths]); ax.set_xlabel("mold-coloured fraction\n(3 leaves; 2D: upper only)", labelpad=1)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1, handlelength=0.8, borderaxespad=0.2); ax.set_xlim(0, max(up + lo) * 1.1)
    save(fig, "demo_c_underside")


def fig_d():
    """Slide version of the downstream result: primary metric only (pooled macro-F1 paired differences, 95% CI)."""
    import csv
    rows = list(csv.DictReader(open(f"{R}/results/downstream_minimal/contrasts.csv")))
    keys = [("D-R", "unedited 3D - real"), ("KB2D-R", "KB 2D - real"), ("GEN3D-R", "generic 3D - real"), ("KB3D-R", "KB 3D - real"), ("KB3D-D", "KB 3D - unedited 3D")]
    cols = {1: BLUE, 4: GOLD, 16: GREEN}; fig, ax = plt.subplots(figsize=(3.1, 1.9))
    for j, (k_, lab) in enumerate(keys):
        for i, k in enumerate((1, 4, 16)):
            r = [x for x in rows if x["contrast"] == k_ and x["metric"] == "macro_f1" and int(x["k"]) == k]
            if not r: continue
            r = r[0]; yy = j + (i - 1) * 0.24
            ax.plot([float(r["lo"]), float(r["hi"])], [yy, yy], color=cols[k], lw=1.4, solid_capstyle="round")
            ax.plot(float(r["est"]), yy, "o", color=cols[k], ms=3, label=f"k = {k}" if j == 0 else None)
    ax.axvline(0, color=GRAY, lw=0.6); ax.axvspan(-0.02, 0.02, color="#d6d5d0", alpha=.5, lw=0)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([l for _, l in keys]); ax.invert_yaxis()
    ax.set_xlabel("paired macro-F1 difference (95% CI)"); ax.legend(frameon=False, loc="lower left", handlelength=0.8, borderaxespad=0.1)
    save(fig, "demo_d_downstream")


if __name__ == "__main__":
    for f in (fig_a, fig_b, fig_c, fig_d):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:
            import traceback; traceback.print_exc()


def write_macros():
    """generated/demo_numbers.tex: every number quoted on the new demo / downstream slides."""
    M = {}; f2 = lambda x: "--" if x is None else f"{x:.2f}"; f1 = lambda x: "--" if x is None else f"{x:.1f}"
    pc = lambda x: "--" if x is None else f"{100 * x:.0f}"
    T = S["tomato"]; vals = lambda m, k: T[m][k]["values"]
    rng_ = lambda v: f"{100 * min(v):.0f}--{100 * max(v):.0f}" if v else "--"
    M.update(TmN=T["_n_specimens"], TmUpReal=rng_(vals("REAL_photo", "mold_on_upper")), TmUpCst=rng_(vals("Cstage_2D_kb", "mold_on_upper")), TmUpC=rng_(vals("C_2D_kb_joint", "mold_on_upper")),
             TmMirrorD=rng_(vals("D", "spot_on_lower")), TmMirrorBL=rng_(vals("BLk", "spot_on_lower")), TmLoBLk=rng_(vals("BLk", "mold_on_lower")), TmUpBLk=rng_(vals("BLk", "mold_on_upper")),
             TmUpBLzero=rng_(vals("BL0", "mold_on_upper")), TmUpF=rng_(vals("F", "mold_on_upper")), TmLoF=rng_(vals("F", "mold_on_lower")),
             TmFootK=rng_(vals("BLk", "mold_footprint_on_spots")), TmFootR=rng_(vals("BLr", "mold_footprint_on_spots")), TmUpBLu=rng_(vals("BLu", "mold_on_upper")),
             TmIouD=f2(T["D"]["input_view_iou"]["mean"]), TmIouBL=f2(T["BLk"]["input_view_iou"]["mean"]))
    G = S["mango"]; M.update(MgN=G["_direction_agreement"]["n"], MgAgree=G["_direction_agreement"]["kb"], MgCtrl=G["_direction_agreement"]["opposite_control_reversed"],
                             MgRollD=f1(G["D"]["roll_deg_median"]["mean"]), MgRollCU=f1(G["CU_kb"]["roll_deg_median"]["mean"]), MgRollCD=f1(G["CD_opposite"]["roll_deg_median"]["mean"]),
                             MgStrainCU=pc(G["CU_kb"]["strain_02L_p95"]["mean"]), MgStrainG=pc(G["G"]["strain_02L_p95"]["mean"]),
                             MgIouD=f2(G["D"]["input_view_iou"]["mean"]), MgIouCU=f2(G["CU_kb"]["input_view_iou"]["mean"]))
    P = S["peach"]; M.update(PcN=P["D"]["input_view_iou"]["n"], PcRollD=", ".join(f"{v:+.0f}" for v in P["D"]["roll_deg_median"]["values"]) or "--",
                             PcEnrJtwo=f2(P["J2"]["placement_enrichment"]["mean"]), PcEnrJtwou=f2(P["J2u"]["placement_enrichment"]["mean"]),
                             PcRedFrac=", ".join(f"{100 * v:.0f}" for v in P["J2"]["reddish_area_frac"]["values"]) or "--",
                             PcStrainJtwo=pc(P["J2"]["strain_02L_p95"]["mean"]), PcStrainJ=pc(P["J"]["strain_02L_p95"]["mean"]),
                             PcIouD=f2(P["D"]["input_view_iou"]["mean"]), PcIouJtwo=f2(P["J2"]["input_view_iou"]["mean"]),
                             PcRollUp=f1(P["J2+curl_up_HYP"]["roll_deg_median"]["mean"]), PcRollDown=f1(P["J2+curl_down_HYP"]["roll_deg_median"]["mean"]))
    C = S["corn"]; hf = S.get("corn_J_heightfield_cm3", {})
    M.update(CnN=C["V"]["undercut_frac"]["n"], CnOnKV=f2(C["V"]["centres_on_kernel_frac"]["mean"]), CnOnKVr=f2(C["Vr"]["centres_on_kernel_frac"]["mean"]),
             CnUndV=f2(C["V"]["undercut_frac"]["mean"]), CnVolV=f1(C["V"]["total_volume_cm3"]["mean"]), CnVolJ=f1(float(np.mean(list(hf.values()))) if hf else None),
             CnIouD=f2(C["D"]["input_view_iou"]["mean"]), CnIouV=f2(C["V"]["input_view_iou"]["mean"]),
             CnAbstain=sum(1 for v in S.get("corn_status", {}).values() if str(v).startswith("ABSTAIN")))
    # Scale-free gall volume (review A2): every cm^3 value comes from the ASSUMED ear length
    # (placement.gall_centres, ear_len_cm=18, no ruler), so V / L^3 removes the assumption
    # exactly. Reported in per mille of (ear length)^3.
    L3 = 18.0 ** 3
    vV = C["V"]["total_volume_cm3"]["mean"]; vJ = float(np.mean(list(hf.values()))) if hf else None
    M.update(CnVolVn=f1(1000 * vV / L3) if vV is not None else "--",
             CnVolJn=f1(1000 * vJ / L3) if vJ is not None else "--")
    ds = json.load(open(f"{R}/results/downstream_minimal/summary.json"))
    for e in ds["families"]["decision"]:
        if e["metric"] != "macro_f1": continue
        tag = {"R": "R", "D": "D"}.get(e["b"]);
        if e["a"] != "KB3D" or tag is None: continue
        k = {1: "one", 4: "four", 16: "sixteen"}[e["k"]]
        M[f"Ds{tag}{k}"] = f"{e['est']:+.3f}"; M[f"Ds{tag}{k}ci"] = f"[{e['lo']:+.3f}, {e['hi']:+.3f}]"
    M["DsDecision"] = str(ds.get("decision", {}).get("verdict", ds.get("decision")))[:60] if not isinstance(ds.get("decision"), str) else ds["decision"][:60]
    tex = ["% generated by src/figures/demo_figs.py from results/demos/summary.json and results/downstream_minimal/summary.json -- do not edit by hand"]
    tex += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
    open(f"{OUT}/demo_numbers.tex", "w").write("\n".join(tex) + "\n"); print("demo macros:", len(M))


if __name__ == "__main__":
    write_macros()
