"""Collect every arm's metrics per specimen, run specimen-level paired statistics (brief s15/s16).

Unit of analysis = specimen (seeds averaged within specimen). Clusters for the bootstrap = disease class.
Arms: D (unedited), E/F (GE, pilot), J (KB geometry), G (generic geometry), K (GE appearance-only, KB), H (J + K-style appearance).
Outputs: results/statistics/arm_metrics.csv, specimen_metrics.csv, comparisons.json, ablation_2x2.json
usage (sam3d env): phase_summary.py [--no-ensure]
"""
import os, sys, json, glob, csv, subprocess
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, f"{R}/src")
from evaluation.stats import cluster_bootstrap_diff, wilcoxon_paired, holm
from metrics.kb_grounding import score as kb_geo
PY = sys.executable; OUT = f"{R}/results/statistics"; os.makedirs(OUT, exist_ok=True)
KEYS = lambda sid: "Citrus__Canker__leaf" if sid == "Citrus__Canker__0004" else "__".join(sid.split("__")[:2])


def sh(cmd):
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ensure(sid):
    """Bring pilot GE outputs (E, F) into results/phaseA/<sid>/<arm>/seed0 and compute the common metrics for every arm."""
    P = f"{R}/results/phaseA/{sid}"; D = f"{R}/results/pilot/{sid}/D"; src = f"{R}/results/pilot/sam3d/{sid}/seed0/splat_ge.ply"; key = KEYS(sid)
    for arm in ("E", "F"):
        pl = f"{R}/results/pilot/{sid}/{arm}/seed0/last.ply"; d = f"{P}/{arm}/seed0"
        if os.path.exists(pl) and not os.path.exists(f"{d}/asset.ply"):
            os.makedirs(d, exist_ok=True); os.symlink(os.path.relpath(pl, d), f"{d}/asset.ply")
    for d in sorted(x for x in glob.glob(f"{P}/*/seed*") if os.path.isdir(x)):
        if not os.path.exists(f"{d}/asset.ply"): continue
        if not os.path.exists(f"{d}/render/rgb/017.png"): sh(f"{PY} {R}/src/rendering/rig_render.py {d}/asset.ply {D}/rig.json {d}/render")
        for name, cmd in [("view_metrics.json", f"{R}/src/metrics/view_metrics.py {P}/D_render {d}/render {D}/rig.json {d}/view_metrics.json"),
                          ("gaussian_diff.json", f"{R}/src/metrics/gaussian_diff.py {src} {d}/asset.ply {d}/gaussian_diff.json"),
                          ("geom_attributes.json", f"{R}/src/metrics/geom_attributes.py {src} {d}/asset.ply {d}/geom_attributes.json"),
                          ("appearance_kb.json", f"{R}/src/metrics/appearance_kb.py {P}/D_render {d}/render {D}/rig.json {key} {d}/appearance_kb.json")]:
            if not os.path.exists(f"{d}/{name}"): sh(f"{PY} {cmd}")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def rows():
    out = []
    for P in sorted(glob.glob(f"{R}/results/phaseA/*__*__*")):
        sid = os.path.basename(P); key = KEYS(sid); cls = "__".join(sid.split("__")[:2])
        out.append(dict(sid=sid, cls=cls, arm="D", seed=0, geometry_recall=0.0 if kb_geo({"distorted": 0, "blister_density": 0, "thickened": 1, "outgrowth": 0, "depression": 0}, key)["required_scorable"] else None,
                        appearance_recall=0.0, appearance_recall_v1=0.0, unsupported_rate=0.0, unsupported_appearance_rate=0.0, coherence=None, jitter=0.0, sil_change=0.0,
                        depth_change=0.0, grain_ratio=1.0, fuzz_ratio=1.0, dE_obj=0.0, crossview_consistency=None))
        for d in sorted(x for x in glob.glob(f"{P}/*/seed*") if os.path.isdir(x)):
            arm, seed = d.split("/")[-2], int(d.split("seed")[-1])
            ga, vm, gd, ak = load(f"{d}/geom_attributes.json"), load(f"{d}/view_metrics.json"), load(f"{d}/gaussian_diff.json"), load(f"{d}/appearance_kb.json")
            if ga is None or vm is None: continue
            g = kb_geo(ga, key); v = vm["summary"]
            out.append(dict(sid=sid, cls=cls, arm=arm, seed=seed, geometry_recall=g["geometry_recall"], unsupported_rate=g["unsupported_rate"],
                            appearance_recall=ak["appearance_recall"] if ak else None,
                            appearance_recall_v1=ak.get("appearance_recall_v1_redder_only") if ak else None, unsupported_appearance_rate=ak["unsupported_appearance_rate"] if ak else None,
                            coherence=gd["disp_coherence_k16"]["median"] if gd and gd.get("disp_coherence_k16") else None, jitter=ga["jitter"],
                            sil_change=v["sil_change"], depth_change=v["depth_change"], grain_ratio=v["grain_ratio"], fuzz_ratio=v["fuzz_ratio"],
                            dE_obj=v["dE_obj"], crossview_consistency=ak.get("crossview_consistency") if ak else None,
                            lower_surface_ratio=ak.get("lower_surface_ratio") if ak else None))
    return out


def specimen_table(R_):
    keys = [k for k in R_[0] if k not in ("sid", "cls", "arm", "seed")]; spec = {}
    for r in R_:
        spec.setdefault((r["sid"], r["cls"], r["arm"]), []).append(r)
    T = []
    for (sid, cls, arm), rs in spec.items():
        t = dict(sid=sid, cls=cls, arm=arm, n_seeds=len(rs))
        for k in keys:
            vals = [x[k] for x in rs if x.get(k) is not None]
            t[k] = float(np.mean(vals)) if vals else None
        T.append(t)
    return T


def compare(T, a, b, metric, higher_better=True):
    by = {}
    for t in T: by.setdefault(t["sid"], {})[t["arm"]] = t
    pairs = [(v[a][metric], v[b][metric], v[a]["cls"]) for v in by.values() if a in v and b in v and v[a][metric] is not None and v[b][metric] is not None]
    if len(pairs) < 2: return dict(a=a, b=b, metric=metric, n=len(pairs), note="insufficient pairs")
    x, y, c = map(np.array, zip(*pairs))
    bs = cluster_bootstrap_diff(x, y, clusters=c); w = wilcoxon_paired(x, y); sb = cluster_bootstrap_diff(x, y)   # specimen-level bootstrap (sensitivity)
    k = int(len(set(c)))
    return dict(a=a, b=b, metric=metric, n=len(pairs), n_classes=k, mean_a=float(x.mean()), mean_b=float(y.mean()), **bs,
                ci95_specimen_bootstrap=sb["ci95"], ci_note=("cluster CI degenerate/unreliable: <3 classes" if k < 3 else ""),
                wilcoxon_p=w.get("p") if isinstance(w, dict) else w, n_positive=int((x > y).sum()), n_negative=int((x < y).sum()), higher_better=higher_better)


def main(ensure_first=True):
    sids = [os.path.basename(p) for p in sorted(glob.glob(f"{R}/results/phaseA/*__*__*"))]
    if ensure_first:
        for s in sids: ensure(s)
    A = rows(); T = specimen_table(A)
    for name, data in (("arm_metrics.csv", A), ("specimen_metrics.csv", T)):
        ks = sorted({k for r in data for k in r}, key=lambda k: (k not in ("sid", "cls", "arm", "seed", "n_seeds"), k))
        with open(f"{OUT}/{name}", "w", newline="") as f:
            w = csv.DictWriter(f, ks); w.writeheader(); [w.writerow(r) for r in data]
    C = [compare(T, "J", "G", "geometry_recall"), compare(T, "J", "G", "unsupported_rate", False), compare(T, "J", "E", "coherence"),
         compare(T, "J", "E", "grain_ratio", False), compare(T, "K", "E", "appearance_recall"), compare(T, "K", "E", "grain_ratio", False),
         compare(T, "K", "E", "fuzz_ratio", False), compare(T, "H", "F", "geometry_recall"), compare(T, "H", "F", "appearance_recall"),
         compare(T, "H", "F", "grain_ratio", False), compare(T, "F", "E", "appearance_recall"), compare(T, "H", "G", "appearance_recall"),
         compare(T, "H", "G", "geometry_recall"), compare(T, "F", "E", "coherence"),
         compare(T, "K", "E", "appearance_recall_v1"), compare(T, "F", "E", "appearance_recall_v1")]
    ps = {f"{c['a']}-{c['b']}:{c['metric']}": c["wilcoxon_p"] for c in C if c.get("wilcoxon_p") is not None and not np.isnan(c["wilcoxon_p"])}
    adj = holm(ps) if ps else {}
    for c in C:
        k = f"{c['a']}-{c['b']}:{c['metric']}"; c["p_holm"] = adj.get(k)
    json.dump(C, open(f"{OUT}/comparisons.json", "w"), indent=1)
    # 2x2: geometry editing x KB conditioning. Outcome = total KB recall (mean of scorable geometry and appearance recall)
    def total(t):
        v = [t[k] for k in ("geometry_recall", "appearance_recall") if t.get(k) is not None]; return float(np.mean(v)) if v else None
    by = {}
    for t in T: by.setdefault(t["sid"], {})[t["arm"]] = total(t)
    cells = {"D (no geo, no KB)": "D", "K (no geo, KB)": "K", "G (geo, no KB)": "G", "H (geo, KB)": "H"}
    full = [v for v in by.values() if all(v.get(x) is not None for x in cells.values())]
    ab = dict(n_specimens_complete=len(full), cells={n: (float(np.mean([v[a] for v in full])) if full else None) for n, a in cells.items()})
    if full:
        inter = [(v["H"] - v["G"]) - (v["K"] - v["D"]) for v in full]; ab["interaction_mean"] = float(np.mean(inter)); ab["interaction_per_specimen"] = inter
    json.dump(ab, open(f"{OUT}/ablation_2x2.json", "w"), indent=1)
    print(json.dumps(dict(n_rows=len(A), n_specimen_rows=len(T), ablation=ab), indent=1))
    for c in C:
        if "diff" in c: print(f"{c['a']} vs {c['b']} {c['metric']}: n={c['n']} ({c['n_classes']} cl) {c['mean_a']:.3f} vs {c['mean_b']:.3f} diff {c['diff']:+.3f} clusterCI [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] specCI [{c['ci95_specimen_bootstrap'][0]:+.3f},{c['ci95_specimen_bootstrap'][1]:+.3f}] +{c['n_positive']}/-{c['n_negative']} p={c['wilcoxon_p']} holm={c['p_holm']} {c['ci_note']}")
        else: print(f"{c['a']} vs {c['b']} {c['metric']}: {c.get('note')} (n={c['n']})")


if __name__ == "__main__":
    main("--no-ensure" not in sys.argv)
