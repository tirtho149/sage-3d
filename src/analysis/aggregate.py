"""Aggregate pilot outputs into per-specimen tables + specimen-level statistics.

Inputs:  results/pilot/sam3d/<sid>/seed*/{e1.json,run_meta.json,splat.ply}
         results/pilot/<sid>/{A,B,C,C_nogeo,C_mismatch,Bmv,Cmv,D,E,F}/...
Outputs: results/pilot/summary/{e1.csv, e1_stability.csv, e2.csv, e4.csv, runtime.csv, stats.json}
Unit of analysis = specimen (15). Renders/views are never counted as independent samples.
PILOT: n is small by design; report effect sizes + CIs, not confirmatory claims.
"""
import os, sys, json, glob
import numpy as np, pandas as pd
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from evaluation import clip_metrics as CM, stats as S, metrics as M

P = f"{ROOT}/results/pilot"; OUT = f"{P}/summary"; os.makedirs(OUT, exist_ok=True)
load = lambda p: np.asarray(Image.open(p).convert("RGB"))
RING = list(range(8)); NB = [(i, (i + 1) % 8) for i in range(8)]


def splat_xyz(p):
    from plyfile import PlyData
    v = PlyData.read(p)["vertex"].data; return np.stack([v["x"], v["y"], v["z"]], 1)


def chamfer_norm(a, b):
    from scipy.spatial import cKDTree
    diag = np.linalg.norm(a.max(0) - a.min(0)); return float((cKDTree(b).query(a)[0].mean() + cKDTree(a).query(b)[0].mean()) / 2 / diag)


specs = sorted(d for d in os.listdir(P) if "__" in d and os.path.isdir(f"{P}/{d}") and not d.startswith("_") and d != "sam3d")
cls = lambda sid: sid.rsplit("__", 1)[0]

# ---------- E1 ----------
rows = []
for f in sorted(glob.glob(f"{P}/sam3d/*/seed*/e1.json")):
    sid, seed = f.split("/")[-3], f.split("/")[-2]
    r = json.load(open(f)); m = json.load(open(os.path.dirname(f) + "/run_meta.json"))
    rows.append(dict(specimen=sid, cls=cls(sid), seed=seed, **{k: r[k] for k in ("mask_iou", "delta_e", "ssim", "lpips")},
                     infer_s=m.get("infer_s"), peak_vram_gb=m.get("peak_vram_gb")))
e1 = pd.DataFrame(rows); e1.to_csv(f"{OUT}/e1.csv", index=False)
stab = []
for d in sorted(glob.glob(f"{P}/sam3d/*")):
    ps = sorted(glob.glob(f"{d}/seed*/splat.ply"))
    if len(ps) > 1:
        xs = [splat_xyz(p) for p in ps]
        stab.append(dict(specimen=os.path.basename(d), cls=cls(os.path.basename(d)), n_seeds=len(ps),
                         chamfer_norm_mean=float(np.mean([chamfer_norm(xs[i], xs[j]) for i in range(len(xs)) for j in range(i + 1, len(xs))]))))
pd.DataFrame(stab).to_csv(f"{OUT}/e1_stability.csv", index=False)

# ---------- E2 / E4 ----------
e2, e4 = [], []
for sid in specs:
    b = f"{P}/{sid}"
    if not os.path.exists(f"{b}/prompts.json"):
        continue
    pr = json.load(open(f"{b}/prompts.json")); pt = pr["prompts"]
    src = f"a photo of a {sid.split('__')[0].lower()} {pr['organ']}"
    kb, mm = pt.get("kb"), pt.get("kb_mismatched")
    inp = load(f"{ROOT}/data/inputs/{sid}/image.png")
    # 2D arms on the input image
    for arm, cond in [("B", "generic"), ("C", "kb"), ("C_nogeo", "kb_no_geometry"), ("C_mismatch", "kb_mismatched")]:
        for f in sorted(glob.glob(f"{b}/{arm}/seed*.png")):
            ed = load(f)
            e2.append(dict(specimen=sid, cls=cls(sid), arm=arm, rep="2D", seed=f.split("seed")[-1][:-4],
                           dir_sim_kb=float(CM.direction_similarity([inp], [ed], src, kb)[0]),
                           kb_specificity=float(CM.kb_specificity([inp], [ed], src, kb, mm)[0]) if mm else np.nan,
                           change_in=M.preservation(inp, ed, np.asarray(Image.open(f"{ROOT}/data/inputs/{sid}/0.png")) > 127)["change_in"]))
    # 3D arms: renders vs unedited renders, same cameras, all 18 views
    dv = sorted(glob.glob(f"{b}/D/images/*.png")); D = [load(x) for x in dv]
    for arm in ("E", "F"):
        vv = sorted(glob.glob(f"{b}/{arm}/seed0/views/images/*.png"))
        if len(vv) != len(dv) or not vv:
            continue
        V = [load(x) for x in vv]
        ds = CM.direction_similarity(D, V, src, kb); ks = CM.kb_specificity(D, V, src, kb, mm) if mm else np.full(len(V), np.nan)
        e2.append(dict(specimen=sid, cls=cls(sid), arm=arm, rep="3D", seed="0", dir_sim_kb=float(ds.mean()), kb_specificity=float(np.nanmean(ks)),
                       dir_sim_kb_bottom=float(ds[-1]), kb_specificity_bottom=float(ks[-1]), dir_sim_kb_ring=float(ds[RING].mean())))
        e4.append(dict(specimen=sid, cls=cls(sid), arm=arm, rep="3D", consistency=CM.directional_consistency([D[i] for i in RING], [V[i] for i in RING], NB)))
    for arm in ("Bmv", "Cmv"):
        vv = [f"{b}/{arm}/{i:03d}.png" for i in RING]
        if all(os.path.exists(x) for x in vv):
            e4.append(dict(specimen=sid, cls=cls(sid), arm=arm, rep="2D-per-view",
                           consistency=CM.directional_consistency([D[i] for i in RING], [load(x) for x in vv], NB)))
e2 = pd.DataFrame(e2); e4 = pd.DataFrame(e4)
e2.to_csv(f"{OUT}/e2.csv", index=False); e4.to_csv(f"{OUT}/e4.csv", index=False)

# ---------- runtime ----------
rt = []
for f in glob.glob(f"{P}/runner_gpu*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if "sec" in r: rt.append(r)
pd.DataFrame(rt).to_csv(f"{OUT}/runtime.csv", index=False)

# ---------- specimen-level paired statistics ----------
st = {}
def paired(df, col, a, b, name):
    x = df[df.arm == a].groupby("specimen")[col].mean(); y = df[df.arm == b].groupby("specimen")[col].mean()
    k = x.index.intersection(y.index)
    if len(k) < 3: st[name] = dict(n=int(len(k)), note="too few specimens"); return
    st[name] = dict(n=int(len(k)), mean_a=float(x[k].mean()), mean_b=float(y[k].mean()),
                    boot=S.cluster_bootstrap_diff(x[k].values, y[k].values, n=5000), wilcoxon=S.wilcoxon_paired(x[k].values, y[k].values))
if len(e2):
    paired(e2, "kb_specificity", "C", "B", "E2_2D_kb_vs_generic_kbspec")
    paired(e2, "kb_specificity", "F", "E", "E2_3D_kb_vs_generic_kbspec")
    paired(e2, "kb_specificity", "F", "C", "E2_3Dkb_vs_2Dkb_kbspec")
    paired(e2, "kb_specificity", "C", "C_mismatch", "E2_2D_kb_vs_mismatched_kbspec")
if len(e4):
    paired(e4, "consistency", "F", "Cmv", "E4_3Dkb_vs_2Dkb_perview_consistency")
    paired(e4, "consistency", "E", "Bmv", "E4_3Dgen_vs_2Dgen_perview_consistency")
json.dump(dict(stats=st, n_specimens_with_e2=int(e2.specimen.nunique()) if len(e2) else 0, pilot=True), open(f"{OUT}/stats.json", "w"), indent=1, default=float)
print(json.dumps(st, indent=1, default=float)[:3000])
