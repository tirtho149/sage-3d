"""Three focused demonstrations (brief v3 items 2-6, 11) + a directional-curl control. Original assets are read-only;
everything is written under results/demos/<demo>/<specimen>/<method>/. All 3D methods are rendered from the SAME
rig views (D rig, views below) and the same corrected input camera.

  peach   Peach Leaf Curl  (0063, 0087, 0006): KB geometry v2 = distort + pucker placed on the photo's reddish/purple
          tissue + thicken (J2) vs the same operators placed uniformly (J2u); curl direction is NOT in the source,
          so curl runs only as two labelled hypotheses (J2+up, J2+down). Measured: observed roll of the
          reconstruction, placement enrichment, tissue strain, relief preservation, input-view IoU vs the real mask.
  mango   Directional control: Mango 0044/0049/0094 leaves with the SAGE phrase for Mango Die Back "Leaves curl
          upwards" (same species and organ; the specimens' own disease is anthracnose) -> CU (KB direction) vs CD
          (opposite direction) vs G (existing generic).
  corn    Corn Smut (0012, 0025, 0052): volumetric galls on kernel tissue (V) vs the same galls placed on random
          Gaussians of the organ (Vr) vs the v1 height-field outgrowth (J) and GaussianEditor (E, F).
  tomato  Tomato Leaf Mold (0015, 0052, 0092): bilayer leaf; mold on the LOWER layer beneath the photo's spots (BLk) vs
          the same mold area at a random lower-layer footprint (BLr) vs mold on the UPPER layer (BLu, negative
          control) vs the bilayer without symptoms (BL0) vs single-sheet D / GaussianEditor E, F, K; 2D KB edits.
usage (sam3d env, GPU): run_demos.py <demo> [specimens comma-separated]
"""
import os, sys, json, time, glob
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, f"{R}/src")
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree
from geometry_editor.anatomy import leaf_frame, ear_frame, xyz, load_vertices, rgb
from geometry_editor import placement, bilayer, gall
from geometry_editor.curl import run_curl, parse
from geometry_editor.deform import run as deform
from rendering.rig_render import render_rig
from metrics.structure3d import roll_metrics, deformation_report, input_view_iou
from metrics import surfaces

VIEWS = [0, 2, 4, 6, 16, 17]
SEEDS = [0, 1, 2]
OUT = f"{R}/results/demos"; LOG = f"{OUT}/execution_log.jsonl"
DEF = json.load(open(f"{R}/configs/geometry/operator_defaults.json")); SPECS = json.load(open(f"{R}/data/kb/structured_specs.json"))


def log(**k):
    os.makedirs(OUT, exist_ok=True); k["t"] = time.strftime("%Y-%m-%d %H:%M:%S"); open(LOG, "a").write(json.dumps(k, default=float) + "\n"); print(json.dumps(k, default=float)[:400], flush=True)


def rig_of(sid):
    return json.load(open(f"{R}/results/pilot/{sid}/D/rig.json"))


def render_set(ply, sid, d, colors=None):
    os.makedirs(f"{d}/views", exist_ok=True)
    for r in render_rig(ply, rig_of(sid), views=VIEWS, colors=colors):
        Image.fromarray(r["rgb"]).save(f"{d}/views/{r['id']:03d}.png")
    iou, rgb_in = input_view_iou(ply, sid); Image.fromarray(rgb_in).save(f"{d}/input_view.png"); return iou


def existing(sid, arm):
    for p in (f"{R}/results/phaseA/{sid}/{arm}/seed0/asset.ply", f"{R}/results/pilot/{sid}/{arm}/seed0/last.ply"):
        if os.path.exists(p): return p
    return None


def matched(F, ply):
    Y = xyz(load_vertices(ply))
    return Y if len(Y) == len(F["X"]) else None


def save_json(o, p):
    json.dump(o, open(p, "w"), indent=1, default=float)


# ------------------------------------------------------------------------------------------------ peach
def demo_peach(sid):
    D = f"{OUT}/peach/{sid}"; os.makedirs(D, exist_ok=True); F = leaf_frame(sid); res = dict(specimen=sid, assumptions=F["assumptions"], methods={})
    np.save(f"{D}/normals.npy", F["n"].astype(np.float32))
    w, plog, frac = placement.pucker_weights(F); np.save(f"{D}/pucker_weights.npy", w.astype(np.float32)); res["placement_log"] = plog
    red = placement.reddish_map(F)[0] >= 0.5; res["reddish_frac"] = float(red.mean())
    ops = SPECS["Peach__Leaf_Curl"]["geometry"]["operators"]; res["kb_operators"] = ops
    res["curl_grounding"] = parse("They usually produce curled leaves at their tips") | dict(note="Ohioline page checked: no blade curl direction (data/kb/supplementary_evidence.json)")

    def ops_v2(weighted):
        out = []
        for g in ops:
            o = dict(DEF[g["op"]], op=g["op"], kb_term=g["term"], kb_phrase=g["phrase"])
            if weighted and g["op"] == "pucker": o["weights_npy"] = f"{D}/pucker_weights.npy"
            out.append(o)
        return out

    def measure(name, ply, extra=None):
        d = f"{D}/{name}"; os.makedirs(d, exist_ok=True); m = dict(input_view_iou=render_set(ply, sid, d))
        Y = matched(F, ply)
        if Y is not None:
            m.update(roll_metrics(F, Y)); m.update(deformation_report(F, Y))
            h = ((Y - F["X"]) * F["n"]).sum(1); from geometry_editor.placement import smooth_on_surface
            hb = h - smooth_on_surface(F["X"], h, k=64, iters=6)               # remove broad warps -> blister band
            e = hb ** 2; m["pucker_energy_on_reddish"] = float(e[red].sum() / (e.sum() + 1e-30)); m["reddish_area_frac"] = float(red.mean())
            m["placement_enrichment"] = m["pucker_energy_on_reddish"] / max(m["reddish_area_frac"], 1e-9)
        if extra: m.update(extra)
        res["methods"][name] = m; log(demo="peach", specimen=sid, method=name, **{k: v for k, v in m.items() if not isinstance(v, (list, dict))})

    measure("D", F["ply"])
    for arm in ("J", "G", "E", "F", "K", "H"):
        p = existing(sid, arm)
        if p: measure(arm, p)
    for s in SEEDS:
        for name, wtd in (("J2", True), ("J2u", False)):
            d = f"{D}/{name}_s{s}"; os.makedirs(d, exist_ok=True)
            if not os.path.exists(f"{d}/asset.ply"):
                deform(F["ply"], dict(seed=s, ops=ops_v2(wtd), iters=300, mode="sheet", normals_npy=f"{D}/normals.npy"), f"{d}/asset.ply")
            measure(f"{name}_s{s}", f"{d}/asset.ply")
    # curl hypotheses on top of J2 seed 0 (direction NOT in the source -> both, labelled)
    F2 = dict(F); F2["ply"] = f"{D}/J2_s0/asset.ply"; F2["X"] = xyz(load_vertices(F2["ply"]))
    for name, sg in (("J2+curl_up_HYP", +1), ("J2+curl_down_HYP", -1)):
        d = f"{D}/{name}"
        if not os.path.exists(f"{d}/asset.ply"): run_curl(F2, sg, 90.0, 0.0, d, name, iters=1000)
        measure(name, f"{d}/asset.ply", dict(hypothesis=True, direction=sg))
    save_json(res, f"{D}/metrics.json"); return res


# ------------------------------------------------------------------------------------------------ mango (direction control)
def demo_mango(sid):
    D = f"{OUT}/mango_curl/{sid}"; os.makedirs(D, exist_ok=True); F = leaf_frame(sid); res = dict(specimen=sid, assumptions=F["assumptions"], methods={})
    ph = "Leaves curl upwards and fall off."; P = parse(ph); res["kb_phrase"] = dict(crop="Mango", disease="Die Back", phrase=ph, parsed=P)
    for name, ply in (("D", F["ply"]), ("G", existing(sid, "G"))):
        if ply is None: continue
        d = f"{D}/{name}"; m = dict(input_view_iou=render_set(ply, sid, d)); Y = matched(F, ply)
        if Y is not None: m.update(roll_metrics(F, Y)); m.update(deformation_report(F, Y))
        res["methods"][name] = m; log(demo="mango", specimen=sid, method=name, **{k: v for k, v in m.items() if not isinstance(v, (list, dict))})
    for s in SEEDS:
        for name, sg in (("CU_kb", P["direction"]), ("CD_opposite", -P["direction"])):
            d = f"{D}/{name}_s{s}"
            if not os.path.exists(f"{d}/asset.ply"): run_curl(F, sg, P["theta_deg"], P["r0"], d, name, seed=s, iters=1000)
            Y = xyz(load_vertices(f"{d}/asset.ply")); m = dict(input_view_iou=render_set(f"{d}/asset.ply", sid, d)); m.update(roll_metrics(F, Y)); m.update(deformation_report(F, Y))
            m["direction_agrees_with_kb"] = bool(np.sign(m["roll_deg_median"] - res["methods"]["D"]["roll_deg_median"]) == P["direction"])
            res["methods"][f"{name}_s{s}"] = m; log(demo="mango", specimen=sid, method=f"{name}_s{s}", **{k: v for k, v in m.items() if not isinstance(v, (list, dict))})
    save_json(res, f"{D}/metrics.json"); return res


# ------------------------------------------------------------------------------------------------ corn
def real_gall_components(E, cm_per_unit):
    """Existing (photo) galls in the reconstruction: connected components of gall-labelled Gaussians (radius graph)."""
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    Xg = E["X"][E["gall"]]
    if len(Xg) < 50: return []
    sub = np.random.default_rng(0).choice(len(Xg), min(len(Xg), 20000), replace=False); Xs = Xg[sub]
    sp = np.median(cKDTree(Xs).query(Xs, k=2)[0][:, 1]); pairs = cKDTree(Xs).query_pairs(3 * sp, output_type="ndarray")
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(Xs),) * 2); n, lab = connected_components(A, directed=False)
    out = []
    for k in range(n):
        P = Xs[lab == k]
        if len(P) < 200: continue
        ext = np.ptp(P @ np.linalg.eigh(np.cov(P.T))[1], 0); out.append(dict(n=int(len(P)), extent_cm=(ext * cm_per_unit).round(2).tolist()))
    return out


def demo_corn(sid):
    D = f"{OUT}/corn/{sid}"; os.makedirs(D, exist_ok=True); E = ear_frame(sid); res = dict(specimen=sid, assumptions=E["assumptions"], methods={})
    res["evidence"] = json.load(open(f"{R}/data/kb/supplementary_evidence.json"))["Corn__Smut"]

    def measure(name, ply, extra=None):
        d = f"{D}/{name}"; m = dict(input_view_iou=render_set(ply, sid, d))
        if extra: m.update(extra)
        res["methods"][name] = m; log(demo="corn", specimen=sid, method=name, **{k: v for k, v in m.items() if not isinstance(v, (list, dict))})

    measure("D", E["ply"])
    for arm in ("J", "G", "E", "F"):
        p = existing(sid, arm)
        if p is None: continue
        Y = xyz(load_vertices(p)); ex = {}
        if len(Y) == len(E["X"]):          # height-field outgrowth: added volume ~ sum of positive normal offsets x area
            h = ((Y - E["X"]) * E["n"]).sum(1); sp = np.median(cKDTree(E["X"][:20000]).query(E["X"][:20000], k=2)[0][:, 1])
            area = sp * sp; ex["added_volume_heightfield"] = float(np.clip(h, 0, None).sum() * area)
            ex["max_offset_rel"] = float(h.max() / E["L"]); ex["undercut_frac"] = 0.0
        measure(arm, p, ex)
    if not E["kernel_ok"]:
        res["status"] = "ABSTAIN: no kernel tissue identifiable in this reconstruction (the evidence-based placement rule cannot be applied)"
        res["real_galls_in_photo"] = real_gall_components(E, 1.0); log(demo="corn", specimen=sid, method="V", status_detail=res["status"])
        save_json(res, f"{D}/metrics.json"); return res
    for s in SEEDS:
        rng = np.random.default_rng(s); C, N, Dm, info, glog = placement.gall_centres(E, n=3, rng=rng)
        cm = info["cm_per_unit"]; res.setdefault("scale", info); res.setdefault("placement_log", glog)
        for name, rand in (("V", False), ("Vr", True)):
            d = f"{D}/{name}_s{s}"; os.makedirs(d, exist_ok=True)
            if rand:                       # same diameters, centres on random Gaussians of the whole organ
                idx = np.random.default_rng(100 + s).choice(len(E["X"]), len(C), replace=False); Cc, Nc = E["X"][idx], E["n"][idx]
            else: Cc, Nc = C, N
            rep = gall.build(E, Cc, Nc, Dm, f"{d}/asset.ply", rng=np.random.default_rng(s))
            tree = cKDTree(E["X"]); lab_at = E["kernel"][tree.query(Cc)[1]]
            vol_cm3 = [g["volume"] * cm ** 3 for g in rep["galls"]]
            ex = dict(n_galls=len(rep["galls"]), centres_on_kernel_frac=float(lab_at.mean()), diam_cm=[g["diameter"] * cm for g in rep["galls"]],
                      eq_diam_cm=[g["eq_diameter"] * cm for g in rep["galls"]], volume_cm3=vol_cm3, total_volume_cm3=float(np.sum(vol_cm3)),
                      undercut_frac=float(np.mean([g["undercut_frac"] for g in rep["galls"]])), kernels_replaced=int(sum(g["kernels_replaced"] for g in rep["galls"])),
                      within_kb_size_range=bool(all(1.0 <= g["diameter"] * cm <= 30.0 for g in rep["galls"])))
            measure(f"{name}_s{s}", f"{d}/asset.ply", ex)
    res["real_galls_in_photo"] = real_gall_components(E, res["scale"]["cm_per_unit"])
    save_json(res, f"{D}/metrics.json"); return res


# ------------------------------------------------------------------------------------------------ tomato
def demo_tomato(sid):
    D = f"{OUT}/tomato/{sid}"; os.makedirs(D, exist_ok=True); F = leaf_frame(sid); rig = rig_of(sid); res = dict(specimen=sid, assumptions=F["assumptions"], methods={})
    m, slog = placement.spot_map(F); res["placement_log"] = slog; res["spot_frac"] = float((m >= 0.5).mean())
    cams = surfaces.camera_centres(rig); V18 = list(range(18))
    d0 = render_rig(F["ply"], rig, views=[0])[0]["rgb"]; fg = d0.astype(int).sum(-1) < 735; _, _, ref = surfaces.classes(d0, fg); res["colour_ref"] = ref

    def stats(ply, layer_ids=None):
        Rr = render_rig(ply, rig, views=V18)
        if layer_ids is not None:
            c = np.where(layer_ids[:, None] == 0, [1.0, 0, 0], [0, 0, 1.0]); Sd = [r["rgb"] for r in render_rig(ply, rig, colors=c, views=V18)]
        else:
            X = xyz(load_vertices(ply)); nn = F["n"][cKDTree(F["X"]).query(X)[1]]
            Sd = [render_rig(ply, rig, colors=surfaces.side_colours_single(X, nn, cams[v]), views=[v])[0]["rgb"] for v in V18]
        return surfaces.surface_stats([r["rgb"] for r in Rr], Sd, ref)

    def measure(name, ply, layer_ids=None, extra=None):
        d = f"{D}/{name}"; mm = dict(input_view_iou=render_set(ply, sid, d)); mm.update(stats(ply, layer_ids))
        if extra: mm.update(extra)
        res["methods"][name] = mm; log(demo="tomato", specimen=sid, method=name, **{k: v for k, v in mm.items() if not isinstance(v, (list, dict))})

    measure("D", F["ply"])
    for arm in ("E", "F", "K"):
        p = existing(sid, arm)
        if p: measure(arm, p)
    for s in SEEDS:
        rng = np.random.default_rng(s); mold = bilayer.mold_colour(F["X"], rng, F["L"])
        # random footprint with the same area: the spot map rotated to a random Gaussian-centred position (same shape statistics)
        k = int((m >= 0.5).sum()); centres = F["X"][np.random.default_rng(400 + s).choice(len(F["X"]), max(1, int(k / 400)), replace=False)]
        dd, _ = cKDTree(centres).query(F["X"]); thr = np.sort(dd)[min(k, len(dd) - 1)]; mr = placement.smooth_on_surface(F["X"], (dd <= thr).astype(float), k=24, iters=2)
        cases = {"BLk": dict(lesion=m, lower_symptom=m, lower_symptom_rgb=mold), "BLr": dict(lesion=m, lower_symptom=mr, lower_symptom_rgb=mold),
                 "BLu": dict(lesion=m, upper_rgb=(1 - m[:, None]) * rgb(F["v"]) + m[:, None] * mold), "BL0": dict(lesion=m)}
        for name, kw in cases.items():
            if name == "BL0" and s > 0: continue
            d = f"{D}/{name}_s{s}"; os.makedirs(d, exist_ok=True); bilayer.build(F, f"{d}/asset.ply", **kw)
            ex = {}
            if "lower_symptom" in kw:
                ls = kw["lower_symptom"]; ex["mold_footprint_on_spots"] = float((ls * (m >= 0.5)).sum() / (ls.sum() + 1e-9))
            measure(f"{name}_s{s}", f"{d}/asset.ply", np.load(f"{d}/asset.ply.layers.npy"), ex)
    # 2D edits of the photo (upper surface only): mold-class fraction inside the organ mask vs the real photo
    mask = np.asarray(Image.open(f"{R}/data/inputs/{sid}/0.png").convert("L")) > 127; real = np.asarray(Image.open(f"{R}/data/inputs/{sid}/image.png").convert("RGB"))
    for name, p in (("REAL_photo", None), ("C_2D_kb_joint", f"{R}/results/pilot/{sid}/C/seed0.png"), ("Cstage_2D_kb", f"{R}/results/phaseA/{sid}/C_stage/seed0.png"),
                    ("B_2D_generic", f"{R}/results/pilot/{sid}/B/seed0.png")):
        im = real if p is None else (np.asarray(Image.open(p).convert("RGB").resize(real.shape[1::-1])) if os.path.exists(p) else None)
        if im is None: continue
        mo, sp, _ = surfaces.classes(im, mask, ref); res["methods"][name] = dict(mold_on_upper=float(mo[mask].mean()), spot_on_upper=float(sp[mask].mean()), surface="upper only (photo)")
        log(demo="tomato", specimen=sid, method=name, **res["methods"][name])
    save_json(res, f"{D}/metrics.json"); return res


if __name__ == "__main__":
    demo = sys.argv[1]
    sids = sys.argv[2].split(",") if len(sys.argv) > 2 else {"peach": ["Peach__Leaf_Curl__0063", "Peach__Leaf_Curl__0087", "Peach__Leaf_Curl__0006"],
                                                              "mango": ["Mango__Anthracnose__0044", "Mango__Anthracnose__0049", "Mango__Anthracnose__0094"],
                                                              "corn": ["Corn__Smut__0012", "Corn__Smut__0025", "Corn__Smut__0052"],
                                                              "tomato": ["Tomato__Leaf_Mold__0015", "Tomato__Leaf_Mold__0052", "Tomato__Leaf_Mold__0092"]}[demo]
    fn = dict(peach=demo_peach, mango=demo_mango, corn=demo_corn, tomato=demo_tomato)[demo]
    for sid in sids:
        t = time.time()
        try:
            fn(sid); log(demo=demo, specimen=sid, status="done", sec=time.time() - t)
        except Exception as e:
            import traceback; traceback.print_exc(); log(demo=demo, specimen=sid, status="FAILED", error=repr(e)[:300])
