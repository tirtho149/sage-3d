"""Minimal downstream real-data falsification test (item 9; pre-registered in reports/14_downstream_minimal.md).

Pipeline (run from the repo root):
  1. ~/Desktop/Research/.venv/bin/python -m src.downstream.prepare --work <scratch>      # select/extract/composite/leakage
  2. CUDA_VISIBLE_DEVICES=0 ~/micromamba/envs/sam3d/bin/python -m src.downstream.features  # DINOv2 features (cached)
  3. ~/Desktop/Research/.venv/bin/python -m src.downstream.minimal_probe                  # this file: probes + stats + figure

Few-shot logistic-regression probes on frozen features; k real images per class (k in {1,4,16}), 50 paired draws;
each synthetic condition adds N=18 images to the pilot class only. Metrics on the real eval set (SAGE test+val,
cluster-disjoint from the build pool). Two-level paired bootstrap (draws x eval images), Holm correction.
"""
import json, warnings, zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

from .common import OUT, CROPS, KS, N_DRAWS, N_BOOT, PRIMARY_CONDS, EXPLORATORY_CONDS

warnings.filterwarnings("ignore")
RENDER_CONDS = ["D", "GEN3D", "KB3D", "J", "F"]
ALL_CONDS = PRIMARY_CONDS + EXPLORATORY_CONDS + [f"{c}_white" for c in RENDER_CONDS]
METRICS = ["macro_f1", "bal_acc", "pilot_recall", "fpr_other"]
DECISION = [("KB3D", "R"), ("KB3D", "D")]
SECONDARY = [(c, "R") for c in ["S", "D", "KB2D", "GEN3D", "KB3D"]]
DESCRIPTIVE = [("KB3D", "KB2D"), ("KB3D", "GEN3D"), ("J", "D"), ("J", "R"), ("F", "R"), ("F", "D"),
               ("GEN3D", "D"), ("KB2D", "S"), ("D", "S"), ("KB3D_white", "D_white"), ("D_white", "R"),
               ("KB3D_white", "R")]


def load():
    z = np.load(OUT / "features.npz")
    fmap = {p: i for i, p in enumerate(z["paths"])}
    F = z["feats"].astype(np.float32)
    real = pd.read_csv(OUT / "image_manifest_real.csv")
    syn = pd.read_csv(OUT / "image_manifest_syn.csv")
    real["fi"] = real.path.map(fmap); syn["fi"] = syn.path.map(fmap)
    return F, real, syn


def crop_data(F, real, syn, crop):
    pilot, others = CROPS[crop]
    classes = [pilot] + others  # pilot class = label 0
    lab = {c: i for i, c in enumerate(classes)}
    r = real[real.crop == crop]
    ev = r[r.role == "eval"]; tr = r[r.role == "train"]
    pools = [tr[tr.disease == c].fi.values for c in classes]
    s = syn[syn.crop == crop]
    synf = {}
    for c in PRIMARY_CONDS + EXPLORATORY_CONDS:
        if c == "R":
            continue
        g = s[(s.cond == c) & (s.bgmode == "natural")]
        if len(g): synf[c] = F[g.fi.values]
    for c in RENDER_CONDS:
        g = s[(s.cond == c) & (s.bgmode == "white")]
        if len(g): synf[f"{c}_white"] = F[g.fi.values]
    return dict(classes=classes, Xev=F[ev.fi.values], yev=ev.disease.map(lab).values, pools=pools, synf=synf)


def fit_predict(X, y, Xev):
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
    clf.fit(X, y)
    return clf.predict(Xev).astype(np.int8)


def run_crop(args):
    crop, F, real, syn = args
    d = crop_data(F, real, syn, crop)
    conds = ["R"] + [c for c in ALL_CONDS if c in d["synf"]]
    preds = {}
    for k in KS:
        P = np.full((len(ALL_CONDS), N_DRAWS, len(d["yev"])), -1, np.int8)
        for draw in range(N_DRAWS):
            rng = np.random.default_rng(zlib.crc32(f"{crop}|{k}|{draw}".encode()))
            idx = [rng.choice(p, size=k, replace=False) for p in d["pools"]]
            Xr = F[np.concatenate(idx)]; yr = np.repeat(np.arange(len(idx)), k)
            for c in conds:
                if c == "R":
                    X, y = Xr, yr
                else:
                    Xs = d["synf"][c]
                    X = np.concatenate([Xr, Xs]); y = np.concatenate([yr, np.zeros(len(Xs), int)])
                P[ALL_CONDS.index(c), draw] = fit_predict(X, y, d["Xev"])
        preds[k] = P
        print(crop, "k", k, "done", flush=True)
    return crop, d["yev"], d["classes"], preds


def metrics_from_conf(C):
    """C: (..., K, K) confusion (rows=true, cols=pred). Returns dict of (...) arrays; pilot = class 0."""
    tp = np.diagonal(C, axis1=-2, axis2=-1).astype(float)
    rows = C.sum(-1).astype(float); cols = C.sum(-2).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(rows > 0, tp / rows, 0.0)
        prec = np.where(cols > 0, tp / cols, 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    neg = rows[..., 1:].sum(-1)
    fp0 = C[..., 1:, 0].sum(-1)
    return dict(macro_f1=f1.mean(-1), bal_acc=rec.mean(-1), pilot_recall=rec[..., 0], fpr_other=fp0 / neg)


def conf(P, y, K):
    """P: (M, n) predictions, y: (n,) -> (M, K, K)."""
    M = P.shape[0]
    flat = (np.arange(M)[:, None] * K + y[None, :]) * K + P.astype(np.int64)
    return np.bincount(flat.ravel(), minlength=M * K * K).reshape(M, K, K)


def boot_crop(args):
    crop, y, K, P, k, seed = args
    nc, nd, n = P.shape
    rng = np.random.default_rng(seed)
    byc = [np.where(y == c)[0] for c in range(K)]
    out = np.full((N_BOOT, nc, len(METRICS)), np.nan)
    valid = (P[:, 0, 0] >= 0)
    Pv = P[valid]
    for b in range(N_BOOT):
        di = rng.integers(nd, size=nd)
        ei = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in byc])
        Pb = Pv[:, di][:, :, ei].reshape(-1, len(ei))
        m = metrics_from_conf(conf(Pb, y[ei], K))
        out[b, valid] = np.stack([m[x].reshape(valid.sum(), nd).mean(1) for x in METRICS], -1)
    return crop, k, out


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p)
    adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * p[i])); adj[i] = run
    return adj


def main():
    F, real, syn = load()
    log = []
    with ProcessPoolExecutor(5) as ex:
        res = list(ex.map(run_crop, [(c, F, real, syn) for c in CROPS]))
    Y = {c: (y, cl) for c, y, cl, _ in res}; PR = {c: p for c, _, _, p in res}
    np.savez_compressed(OUT / "predictions.npz", **{f"{c}|{k}": PR[c][k] for c in PR for k in KS},
                        **{f"{c}|y": Y[c][0] for c in Y}, conds=np.array(ALL_CONDS))
    # per-draw metrics
    rows = []
    for c in CROPS:
        y, cl = Y[c]
        for k in KS:
            P = PR[c][k]
            for ci, cond in enumerate(ALL_CONDS):
                if P[ci, 0, 0] < 0: continue
                m = metrics_from_conf(conf(P[ci], y, len(cl)))
                for dr in range(N_DRAWS):
                    rows.append(dict(crop=c, k=k, draw=dr, cond=cond, **{x: float(m[x][dr]) for x in METRICS}))
    pdraw = pd.DataFrame(rows); pdraw.to_csv(OUT / "per_draw_metrics.csv", index=False)
    # bootstrap
    jobs = [(c, Y[c][0], len(Y[c][1]), PR[c][k], k, zlib.crc32(f"boot|{c}|{k}".encode())) for c in CROPS for k in KS]
    with ProcessPoolExecutor(10) as ex:
        B = {(c, k): o for c, k, o in ex.map(boot_crop, jobs)}
    point = pdraw.groupby(["crop", "k", "cond"])[METRICS].mean()

    def contrast(a, b, k, crops=None, metric="macro_f1"):
        crops = [c for c in (crops or list(CROPS)) if not np.isnan(B[(c, k)][0, ALL_CONDS.index(a), 0])
                 and not np.isnan(B[(c, k)][0, ALL_CONDS.index(b), 0])]
        mi = METRICS.index(metric)
        per = {c: point.loc[(c, k, a), metric] - point.loc[(c, k, b), metric] for c in crops}
        bd = np.mean([B[(c, k)][:, ALL_CONDS.index(a), mi] - B[(c, k)][:, ALL_CONDS.index(b), mi] for c in crops], 0)
        est = float(np.mean(list(per.values())))
        lo, hi = np.percentile(bd, [2.5, 97.5])
        p = float(min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean())))
        pc = {}
        for c in crops:
            bc = B[(c, k)][:, ALL_CONDS.index(a), mi] - B[(c, k)][:, ALL_CONDS.index(b), mi]
            pc[c] = dict(est=float(per[c]), lo=float(np.percentile(bc, 2.5)), hi=float(np.percentile(bc, 97.5)))
        return dict(a=a, b=b, k=k, metric=metric, n_crops=len(crops), est=est, lo=float(lo), hi=float(hi), p=p, per_crop=pc)

    summary = dict(conds=ALL_CONDS, metrics=METRICS, n_draws=N_DRAWS, n_boot=N_BOOT, ks=KS, families={})
    allrows = []
    for fam, pairs in (("decision", DECISION), ("secondary", SECONDARY), ("descriptive", DESCRIPTIVE)):
        res_f = []
        for metric in METRICS:
            rr = [contrast(a, b, k, metric=metric) for a, b in pairs for k in KS]
            if fam != "descriptive":
                adj = holm([r["p"] for r in rr])
                for r, q in zip(rr, adj): r["p_holm"] = float(q)
            res_f += rr
        summary["families"][fam] = res_f
        for r in res_f:
            allrows.append(dict(family=fam, contrast=f"{r['a']}-{r['b']}", k=r["k"], metric=r["metric"], n_crops=r["n_crops"],
                                est=r["est"], lo=r["lo"], hi=r["hi"], p=r["p"], p_holm=r.get("p_holm", np.nan),
                                **{f"{c}_est": v["est"] for c, v in r["per_crop"].items()}))
    ctab = pd.DataFrame(allrows); ctab.to_csv(OUT / "contrasts.csv", index=False)
    # absolute levels (pooled over crops) with bootstrap CIs
    lev = []
    for k in KS:
        for cond in ALL_CONDS:
            crops = [c for c in CROPS if not np.isnan(B[(c, k)][0, ALL_CONDS.index(cond), 0])]
            for mi, metric in enumerate(METRICS):
                bd = np.mean([B[(c, k)][:, ALL_CONDS.index(cond), mi] for c in crops], 0)
                lev.append(dict(k=k, cond=cond, metric=metric, n_crops=len(crops),
                                est=float(np.mean([point.loc[(c, k, cond), metric] for c in crops])),
                                lo=float(np.percentile(bd, 2.5)), hi=float(np.percentile(bd, 97.5))))
    pd.DataFrame(lev).to_csv(OUT / "levels_pooled.csv", index=False)
    point.reset_index().to_csv(OUT / "levels_per_crop.csv", index=False)
    # pre-registered decision
    dec = [r for r in summary["families"]["decision"] if r["metric"] == "macro_f1"]
    supported_k = [k for k in KS if all(any(r["a"] == "KB3D" and r["b"] == b and r["k"] == k and r["est"] > 0 and r["p_holm"] < .05
                                            for r in dec) for b in ("R", "D"))]
    kd_hi = {r["k"]: r["hi"] for r in dec if r["b"] == "D"}
    if supported_k: verdict = "SUPPORTED"
    elif all(v < 0.02 for v in kd_hi.values()): verdict = "NOT SUPPORTED (falsified at this scale)"
    else: verdict = "INCONCLUSIVE"
    summary["decision"] = dict(verdict=verdict, supported_k=supported_k, kb3d_minus_D_upper_ci=kd_hi,
                               rule="supported iff KB3D-R>0 and KB3D-D>0 (pooled macro-F1, Holm p<.05 over 6 tests) at some k; "
                                    "not supported iff additionally upper CI of KB3D-D < +0.02 at every k; else inconclusive")
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print("VERDICT", verdict, kd_hi)
    figure(ctab)


def figure(ctab):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("S-R", "S − R"), ("D-R", "D − R"), ("KB2D-R", "KB2D − R"), ("GEN3D-R", "GEN3D − R"),
            ("KB3D-R", "KB3D − R"), ("KB3D-D", "KB3D − D"), ("KB3D-KB2D", "KB3D − KB2D"), ("KB3D_white-D_white", "KB3D − D (white bg)")]
    cols = {1: "#2a78d6", 4: "#eb6834", 16: "#1baf7a"}; mk = {1: "o", 4: "s", 16: "D"}
    mets = [("macro_f1", "Macro-F1 (primary)"), ("bal_acc", "Balanced accuracy"), ("pilot_recall", "Pilot-class recall"),
            ("fpr_other", "FPR on other classes")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    for ax, (m, title) in zip(axs, mets):
        for j, (key, lab) in enumerate(rows):
            g = ctab[(ctab.contrast == key) & (ctab.metric == m)].drop_duplicates(["k"])
            for i, k in enumerate(KS):
                r = g[g.k == k]
                if not len(r): continue
                r = r.iloc[0]; yy = j + (i - 1) * 0.22
                ax.plot([r.lo, r.hi], [yy, yy], color=cols[k], lw=2, solid_capstyle="round")
                ax.plot(r.est, yy, mk[k], color=cols[k], ms=6, mec="white", mew=1, label=f"k = {k}" if j == 0 else None)
        ax.axvline(0, color="#52514e", lw=1)
        if m == "macro_f1": ax.axvspan(-0.02, 0.02, color="#d6d5d0", alpha=.5, lw=0)
        ax.set_title(title, fontsize=10); ax.grid(axis="x", color="#e4e3df", lw=.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.set_xlabel("paired difference (pooled over crops)", fontsize=8)
    axs[0].set_yticks(range(len(rows))); axs[0].set_yticklabels([l for _, l in rows]); axs[0].invert_yaxis()
    axs[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Minimal downstream test: paired differences on independent real SAGE photos (95% bootstrap CI; grey band = ±0.02 macro-F1)", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"paired_differences.{ext}", dpi=200)


if __name__ == "__main__":
    main()
