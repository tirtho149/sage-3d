"""Inter-rater agreement and method-effect analysis for the expert-review package.

STATUS: no real expert ratings exist yet (2026-10-14). The `simulate` sub-command produces SIMULATED ratings for
testing the pipeline only; every output carries a data_status line saying which kind of data it was computed on.

Two separate questions, reported in separate sections:
  1. AGREEMENT  (is the rubric applied consistently?)  Krippendorff's alpha (ordinal) per criterion with a
     unit-bootstrap CI, quadratic-weighted Cohen's kappa per rater pair, ICC(2,k), intra-rater test-retest on
     duplicated items, catch-item pass rates.
  2. METHOD EFFECT (do methods differ?)  per-method mean of item scores (mean over raters), specimen-clustered
     bootstrap CIs, paired family comparisons over specimens (bootstrap CI + exact sign-flip permutation p, Holm
     across the planned comparisons for the primary criterion).
Method-effect results are only interpretable if agreement on that criterion reaches the pre-registered threshold
(rubric.json analysis.agreement_threshold_for_method_claims); the report states this per criterion.

usage:
  python src/review/agreement.py simulate --key results/review_package_v1/key_DO_NOT_SHARE.csv \
         --out results/review_package_v1/simulated_ratings --n-raters 3 --seed 1
  python src/review/agreement.py analyse --key results/review_package_v1/key_DO_NOT_SHARE.csv \
         --ratings results/review_package_v1/simulated_ratings --out results/review_package_v1/analysis_SIMULATED
"""
import argparse
import csv
import glob
import itertools
import json
import os
import random

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CRITERIA = ["a_symptom_match", "b_distribution", "c_geometry", "d_surface", "e_artefacts", "f_overall_accept"]
BINARY = ["prohibited_feature"]
NO_REAL = "No real expert ratings exist yet (as of 2026-10-14)."


# ================================================================================================ statistics
def _delta2(values, n_g, level):
    """Squared difference function delta^2(c, k) for the sorted distinct values (Krippendorff 2011)."""
    v = np.asarray(values, float)
    V = len(v)
    if level == "nominal":
        return 1.0 - np.eye(V)
    if level == "interval":
        return (v[:, None] - v[None, :]) ** 2
    if level == "ratio":
        s = v[:, None] + v[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            d = ((v[:, None] - v[None, :]) / s) ** 2
        return np.nan_to_num(d)
    if level == "ordinal":
        cum = np.concatenate([[0.0], np.cumsum(n_g)])
        d = np.zeros((V, V))
        for c in range(V):
            for k in range(V):
                lo, hi = min(c, k), max(c, k)
                d[c, k] = (cum[hi + 1] - cum[lo] - (n_g[c] + n_g[k]) / 2.0) ** 2
        return d
    raise ValueError(level)


def coincidence(data):
    """data: raters x units array, np.nan = missing. Returns (values, o) with o the coincidence matrix."""
    data = np.asarray(data, float)
    vals = np.unique(data[~np.isnan(data)])
    idx = {v: i for i, v in enumerate(vals)}
    o = np.zeros((len(vals), len(vals)))
    for u in range(data.shape[1]):
        col = data[:, u]
        col = col[~np.isnan(col)]
        m = len(col)
        if m < 2:
            continue
        cnt = np.zeros(len(vals))
        for x in col:
            cnt[idx[x]] += 1
        # ordered pairs of values from different raters: cnt_c*cnt_k for c!=k, cnt_c*(cnt_c-1) for c==k
        o += (np.outer(cnt, cnt) - np.diag(cnt)) / (m - 1)
    return vals, o


def krippendorff_alpha(data, level="ordinal"):
    """Krippendorff's alpha for a raters x units matrix with np.nan for missing values."""
    vals, o = coincidence(data)
    n_g = o.sum(1)
    n = n_g.sum()
    if n < 2 or len(vals) < 2:
        return float("nan")
    d2 = _delta2(vals, n_g, level)
    Do = (o * d2).sum() / n
    De = (np.outer(n_g, n_g) * d2).sum() / (n * (n - 1))
    return float(1.0 - Do / De) if De > 0 else float("nan")


def alpha_bootstrap(data, level="ordinal", reps=2000, seed=0, ci=0.95):
    """Percentile CI from resampling units (items) with replacement. (Krippendorff's own bootstrap resamples
    pairable values; the unit bootstrap used here respects the item as the sampling unit.)"""
    data = np.asarray(data, float)
    rng = np.random.default_rng(seed)
    U = data.shape[1]
    est = krippendorff_alpha(data, level)
    bs = []
    for _ in range(reps):
        a = krippendorff_alpha(data[:, rng.integers(0, U, U)], level)
        if not np.isnan(a):
            bs.append(a)
    if not bs:
        return est, float("nan"), float("nan")
    lo, hi = np.percentile(bs, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return est, float(lo), float(hi)


def weighted_kappa(x, y, categories=(1, 2, 3, 4, 5), weights="quadratic"):
    """Cohen's weighted kappa for two raters (pairs with a missing value are dropped)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return float("nan")
    cats = list(categories)
    K = len(cats)
    pos = {c: i for i, c in enumerate(cats)}
    O = np.zeros((K, K))
    for a, b in zip(x, y):
        O[pos[a], pos[b]] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    i, j = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    if weights == "quadratic":
        W = ((i - j) / (K - 1)) ** 2
    elif weights == "linear":
        W = np.abs(i - j) / (K - 1)
    else:
        W = (i != j).astype(float)
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else float("nan")


def icc(Y):
    """Shrout & Fleiss (1979) ICCs for a complete units x raters matrix."""
    Y = np.asarray(Y, float)
    n, k = Y.shape
    gm = Y.mean()
    ss_r = k * ((Y.mean(1) - gm) ** 2).sum()
    ss_c = n * ((Y.mean(0) - gm) ** 2).sum()
    ss_t = ((Y - gm) ** 2).sum()
    ss_e = ss_t - ss_r - ss_c
    BMS = ss_r / (n - 1)
    JMS = ss_c / (k - 1)
    EMS = ss_e / ((n - 1) * (k - 1))
    WMS = (ss_c + ss_e) / (n * (k - 1))
    return {"ICC(1,1)": (BMS - WMS) / (BMS + (k - 1) * WMS),
            "ICC(2,1)": (BMS - EMS) / (BMS + (k - 1) * EMS + k * (JMS - EMS) / n),
            "ICC(3,1)": (BMS - EMS) / (BMS + (k - 1) * EMS),
            "ICC(1,k)": (BMS - WMS) / BMS,
            "ICC(2,k)": (BMS - EMS) / (BMS + (JMS - EMS) / n),
            "ICC(3,k)": (BMS - EMS) / BMS, "n_units": n, "k_raters": k}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    m = len(p)
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        adj[i] = run
    return adj.tolist()


def signflip_p(d, max_exact=16, reps=20000, seed=0):
    """Two-sided paired sign-flip permutation p-value for mean(d) = 0 (exact when len(d) <= max_exact)."""
    d = np.asarray(d, float)
    n = len(d)
    if n == 0:
        return float("nan")
    obs = abs(d.mean())
    if n <= max_exact:
        signs = np.array(list(itertools.product([1, -1], repeat=n)))
    else:
        signs = np.random.default_rng(seed).choice([1, -1], size=(reps, n))
    stats = np.abs((signs * d).mean(1))
    return float((stats >= obs - 1e-12).mean())


# ================================================================================================ data
def read_key(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def _num(v):
    v = (v or "").strip()
    if v == "" or v.upper() == "NA":
        return np.nan
    return float(v)


def read_ratings(paths):
    """Returns list of dict rows (rater_id, item_id, criteria as floats/nan, simulated flag)."""
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "*.csv"))) if os.path.isdir(p) else [p]
    rows = []
    for fp in files:
        with open(fp, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("item_id") or not r.get("rater_id"):
                    continue
                out = dict(rater_id=r["rater_id"], item_id=r["item_id"], file=os.path.basename(fp),
                           simulated=str(r.get("simulated", "")).strip() == "1" or r["rater_id"].startswith("SIM"),
                           comment=r.get("comment", ""))
                for c in CRITERIA + BINARY:
                    out[c] = _num(r.get(c))
                rows.append(out)
    return rows


def data_status(rows, confirm_real):
    if not rows:
        return "NO RATINGS LOADED. " + NO_REAL
    if any(r["simulated"] for r in rows):
        return "SIMULATED RATINGS - NOT EXPERT DATA. " + NO_REAL
    if not confirm_real:
        return ("Ratings of unconfirmed origin (run with --confirm-real only for returned expert files). " + NO_REAL)
    return "Expert ratings (confirmed real by the analyst via --confirm-real)."


# ================================================================================================ analysis
def matrix(rows, key, crit, item_filter):
    raters = sorted({r["rater_id"] for r in rows})
    items = sorted({r["item_id"] for r in rows if r["item_id"] in key and item_filter(key[r["item_id"]])})
    M = np.full((len(raters), len(items)), np.nan)
    ri = {r: i for i, r in enumerate(raters)}
    ii = {it: j for j, it in enumerate(items)}
    for r in rows:
        if r["item_id"] in ii:
            M[ri[r["rater_id"]], ii[r["item_id"]]] = r[crit]
    return raters, items, M


def is_primary_unit(k):
    return k["role"] == "test" and k["is_duplicate"] in ("0", "", "False")


def agreement_section(rows, key, reps, seed):
    out = {"per_criterion": {}, "pairwise_kappa": [], "icc": {}, "intra_rater": {}, "catch": {}}
    for c in CRITERIA + BINARY:
        raters, items, M = matrix(rows, key, c, is_primary_unit)
        level = "nominal" if c in BINARY else "ordinal"
        est, lo, hi = alpha_bootstrap(M, level, reps=reps, seed=seed)
        out["per_criterion"][c] = dict(alpha=est, ci95=[lo, hi], level=level, n_items=len(items),
                                       n_raters=len(raters),
                                       n_pairable_values=int(sum((~np.isnan(M[:, u])).sum() for u in range(M.shape[1])
                                                                 if (~np.isnan(M[:, u])).sum() >= 2)))
        if c in CRITERIA:
            for (i, a), (j, b) in itertools.combinations(enumerate(raters), 2):
                out["pairwise_kappa"].append(dict(criterion=c, rater_a=a, rater_b=b,
                                                  kappa_quadratic=weighted_kappa(M[i], M[j]),
                                                  n_items=int((~np.isnan(M[i]) & ~np.isnan(M[j])).sum())))
            comp = M[:, ~np.isnan(M).any(0)].T
            if comp.shape[0] >= 3 and comp.shape[1] >= 2:
                ic = icc(comp)
                out["icc"][c] = {"ICC(2,k)": ic["ICC(2,k)"], "ICC(2,1)": ic["ICC(2,1)"], "n_complete_items":
                                 ic["n_units"], "k_raters": ic["k_raters"]}
    # intra-rater: duplicated items (first vs second presentation)
    dup_pairs = [(k["duplicate_of_item_id"], k["item_id"]) for k in key.values() if k["is_duplicate"] == "1"]
    by = {(r["rater_id"], r["item_id"]): r for r in rows}
    for rater in sorted({r["rater_id"] for r in rows}):
        res = {}
        for c in CRITERIA:
            x, y = [], []
            for a, b in dup_pairs:
                if (rater, a) in by and (rater, b) in by:
                    x.append(by[(rater, a)][c])
                    y.append(by[(rater, b)][c])
            x, y = np.array(x, float), np.array(y, float)
            ok = ~(np.isnan(x) | np.isnan(y))
            res[c] = dict(n_pairs=int(ok.sum()), kappa_quadratic=weighted_kappa(x, y),
                          exact_agreement=float((x[ok] == y[ok]).mean()) if ok.any() else float("nan"),
                          mean_abs_diff=float(np.abs(x[ok] - y[ok]).mean()) if ok.any() else float("nan"))
        out["intra_rater"][rater] = res
    # catch items
    catch_ids = [k["item_id"] for k in key.values() if k["role"] == "catch"]
    rule = RUBRIC["catch_rule"]
    for rater in sorted({r["rater_id"] for r in rows}):
        passes = []
        for it in catch_ids:
            r = by.get((rater, it))
            if r is None:
                continue
            passes.append(bool((r[rule["criterion"]] <= rule["pass_if_at_most"]) or r[rule["or_binary"]] == 1))
        rate = float(np.mean(passes)) if passes else float("nan")
        out["catch"][rater] = dict(n=len(passes), passed=int(sum(passes)), pass_rate=rate,
                                   flagged=bool(passes) and rate < rule["rater_flag_if_pass_rate_below"])
    return out


def item_scores(rows, key, crit, raters_keep):
    """Mean over raters per (first-presentation, non-catch) item."""
    acc = {}
    for r in rows:
        k = key.get(r["item_id"])
        if k is None or not is_primary_unit(k) or r["rater_id"] not in raters_keep or np.isnan(r[crit]):
            continue
        acc.setdefault(r["item_id"], []).append(r[crit])
    return {it: float(np.mean(v)) for it, v in acc.items()}


def cluster_boot_mean(spec_to_vals, reps, rng):
    specs = sorted(spec_to_vals)
    if not specs:
        return float("nan"), float("nan"), float("nan")
    est = float(np.mean([v for s in specs for v in spec_to_vals[s]]))
    bs = []
    for _ in range(reps):
        pick = rng.integers(0, len(specs), len(specs))
        vals = [v for i in pick for v in spec_to_vals[specs[i]]]
        bs.append(np.mean(vals))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi)


def method_section(rows, key, reps, seed, raters_keep, agreement):
    an = RUBRIC["analysis"]
    rng = np.random.default_rng(seed)
    out = {"per_method": [], "comparisons": []}
    thr = an["agreement_threshold_for_method_claims"]
    for c in CRITERIA:
        sc = item_scores(rows, key, c, raters_keep)
        by_m = {}
        for it, v in sc.items():
            k = key[it]
            by_m.setdefault((k["method"], k["method_family"]), {}).setdefault(k["specimen"], []).append(v)
        for (m, fam), d in sorted(by_m.items()):
            est, lo, hi = cluster_boot_mean(d, reps, rng)
            out["per_method"].append(dict(criterion=c, method=m, family=fam, mean=est, ci95=[lo, hi],
                                          n_items=sum(len(v) for v in d.values()), n_specimens=len(d)))
        # family means per specimen
        fam_spec = {}
        for it, v in sc.items():
            k = key[it]
            fam_spec.setdefault(k["method_family"], {}).setdefault(k["specimen"], []).append(v)
        comps = []
        for fa, fb in an["planned_comparisons"]:
            A, B = fam_spec.get(fa, {}), fam_spec.get(fb, {})
            common = sorted(set(A) & set(B))
            d = np.array([np.mean(A[s]) - np.mean(B[s]) for s in common])
            if len(d) == 0:
                comps.append(dict(criterion=c, a=fa, b=fb, n_specimens=0))
                continue
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            comps.append(dict(criterion=c, a=fa, b=fb, n_specimens=len(d), mean_diff=float(d.mean()),
                              ci95=[float(lo), float(hi)], p_signflip=signflip_p(d)))
        if c == an["primary_criterion"]:
            ps = [x.get("p_signflip", np.nan) for x in comps]
            valid = [i for i, p in enumerate(ps) if not np.isnan(p)]
            adj = holm([ps[i] for i in valid]) if valid else []
            for i, a in zip(valid, adj):
                comps[i]["p_holm"] = a
            for x in comps:
                x["status"] = "primary (confirmatory)"
        else:
            for x in comps:
                x["status"] = "secondary (exploratory, unadjusted)"
        alpha_c = agreement["per_criterion"][c]["alpha"]
        for x in comps:
            x["agreement_alpha"] = alpha_c
            x["interpretable"] = bool(not np.isnan(alpha_c) and alpha_c >= thr)
        out["comparisons"] += comps
    return out


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_report(res, out):
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "agreement_report.json"), "w"), indent=1, default=float)
    st = res["data_status"]
    with open(os.path.join(out, "per_method_scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["data_status", "criterion", "method", "family", "mean", "ci95_lo", "ci95_hi", "n_items",
                    "n_specimens"])
        for r in res["method_effect"]["per_method"]:
            w.writerow([st, r["criterion"], r["method"], r["family"], fmt(r["mean"]), fmt(r["ci95"][0]),
                        fmt(r["ci95"][1]), r["n_items"], r["n_specimens"]])
    with open(os.path.join(out, "pairwise_kappa.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["data_status", "criterion", "rater_a", "rater_b", "kappa_quadratic", "n_items"])
        for r in res["agreement"]["pairwise_kappa"]:
            w.writerow([st, r["criterion"], r["rater_a"], r["rater_b"], fmt(r["kappa_quadratic"]), r["n_items"]])
    A = res["agreement"]
    L = ["# Expert-review analysis", "", f"**Data status: {st}**", "",
         f"Raters: {', '.join(res['raters'])}; flagged by catch rule (excluded from the main method-effect analysis, "
         f"kept for agreement): {res['raters_excluded'] or 'none'}.", "",
         "## 1. Agreement (consistency of the rubric; says nothing about which method is better)", "",
         "| criterion | Krippendorff alpha | 95% CI (unit bootstrap) | ICC(2,k) | items |", "|---|---|---|---|---|"]
    for c, v in A["per_criterion"].items():
        ic = A["icc"].get(c, {}).get("ICC(2,k)")
        L.append(f"| {c} ({v['level']}) | {fmt(v['alpha'])} | [{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}] | "
                 f"{fmt(ic) if ic is not None else '-'} | {v['n_items']} |")
    L += ["", "Mean pairwise quadratic-weighted kappa per criterion:", ""]
    for c in CRITERIA:
        ks = [r["kappa_quadratic"] for r in A["pairwise_kappa"] if r["criterion"] == c]
        L.append(f"- {c}: {fmt(float(np.nanmean(ks)) if ks else float('nan'))} over {len(ks)} pairs")
    L += ["", "Intra-rater (duplicated items, quadratic kappa / exact agreement / n pairs):", ""]
    for rater, d in A["intra_rater"].items():
        L.append(f"- {rater}: " + "; ".join(f"{c[:1]} {fmt(v['kappa_quadratic'], 2)}/{fmt(v['exact_agreement'], 2)}"
                                             f"/{v['n_pairs']}" for c, v in d.items()))
    L += ["", "Catch items (pass = a_symptom_match <= 2 or prohibited_feature = yes):", ""]
    for rater, d in A["catch"].items():
        L.append(f"- {rater}: {d['passed']}/{d['n']} passed{' (FLAGGED)' if d['flagged'] else ''}")
    L += ["", "## 2. Method effect (separate question; only interpretable where agreement alpha >= "
          f"{RUBRIC['analysis']['agreement_threshold_for_method_claims']})", "",
          "Planned paired comparisons over specimens (family mean per specimen, A minus B):", "",
          "| criterion | A vs B | n spec | mean diff | 95% CI | p (sign-flip) | p Holm | alpha ok | status |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in res["method_effect"]["comparisons"]:
        if not x.get("n_specimens"):
            L.append(f"| {x['criterion']} | {x['a']} vs {x['b']} | 0 | - | - | - | - | - | no common specimens |")
            continue
        L.append(f"| {x['criterion']} | {x['a']} vs {x['b']} | {x['n_specimens']} | {fmt(x['mean_diff'])} | "
                 f"[{fmt(x['ci95'][0])}, {fmt(x['ci95'][1])}] | {fmt(x['p_signflip'])} | "
                 f"{fmt(x.get('p_holm', float('nan')))} | {'yes' if x['interpretable'] else 'no'} | {x['status']} |")
    if res["raters_excluded"]:
        L += ["", "Sensitivity (all raters, including flagged): primary criterion", ""]
        for x in res["method_effect_all_raters"]["comparisons"]:
            if x["criterion"] == RUBRIC["analysis"]["primary_criterion"] and x.get("n_specimens"):
                L.append(f"- {x['a']} vs {x['b']}: diff {fmt(x['mean_diff'])} [{fmt(x['ci95'][0])}, "
                         f"{fmt(x['ci95'][1])}], p Holm {fmt(x.get('p_holm', float('nan')))}")
    L += ["", "Per-method means are in per_method_scores.csv (specimen-clustered bootstrap CIs).", "",
          f"**Data status: {st}**", ""]
    open(os.path.join(out, "agreement_report.md"), "w").write("\n".join(L))


def analyse(key_path, rating_paths, out, reps=None, seed=None, confirm_real=False, exclude_flagged=True):
    key = read_key(key_path)
    rows = read_ratings(rating_paths)
    unknown = {r["item_id"] for r in rows} - set(key)
    rows = [r for r in rows if r["item_id"] in key]
    an = RUBRIC["analysis"]
    reps = reps or an["bootstrap_reps"]
    seed = an["bootstrap_seed"] if seed is None else seed
    ag = agreement_section(rows, key, reps, seed)
    raters = sorted({r["rater_id"] for r in rows})
    excluded = [r for r, d in ag["catch"].items() if d["flagged"]] if exclude_flagged else []
    keep = set(raters) - set(excluded)
    me_all = method_section(rows, key, reps, seed, set(raters), ag)
    me = method_section(rows, key, reps, seed, keep, ag) if excluded and len(keep) >= 1 else me_all
    res = dict(data_status=data_status(rows, confirm_real), key=key_path, ratings=rating_paths, raters=raters,
               raters_excluded=excluded, raters_used_for_method_effect=sorted(keep), unknown_item_ids=sorted(unknown),
               agreement=ag, method_effect=me, method_effect_all_raters=me_all)
    write_report(res, out)
    return res


# ================================================================================================ simulation
def simulate(key_path, out, n_raters=3, seed=1, noise=0.6, rater_bias_sd=0.3, method_effects=None):
    """SIMULATED ratings for pipeline testing only. Latent item quality = specimen effect + method effect;
    each rater adds a bias and Gaussian noise; values are rounded and clipped to 1..5."""
    key = read_key(key_path)
    rng = np.random.default_rng(seed)
    method_effects = method_effects or {}
    specs = sorted({k["specimen"] for k in key.values()})
    s_eff = {s: rng.normal(0, 0.4) for s in specs}
    latent = {}
    for k in key.values():
        if k["item_key"] in latent:
            continue
        base = 1.4 if k["role"] == "catch" else 3.0 + method_effects.get(k["method"], 0.0) + s_eff[k["specimen"]]
        latent[k["item_key"]] = {c: base + rng.normal(0, 0.3) for c in CRITERIA}
    os.makedirs(out, exist_ok=True)
    files = []
    for r in range(n_raters):
        rid = f"SIM{r + 1:02d}"
        bias = rng.normal(0, rater_bias_sd)
        fp = os.path.join(out, f"ratings_SIMULATED_{rid}.csv")
        with open(fp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["package_id", "rater_id", "item_id", "page"] + CRITERIA + BINARY +
                       ["comment", "seconds_on_item", "saved_at", "simulated"])
            for k in sorted(key.values(), key=lambda k: int(k["page"])):
                lat = latent[k["item_key"]]
                sc = [int(np.clip(np.rint(lat[c] + bias + rng.normal(0, noise)), 1, 5)) for c in CRITERIA]
                pro = int(sc[0] <= 2 and rng.random() < 0.7)
                w.writerow(["sim", rid, k["item_id"], k["page"]] + sc + [pro, "SIMULATED", 30, "", 1])
        files.append(fp)
    return files


def _load_rubric(path):
    global RUBRIC
    RUBRIC = json.load(open(path if os.path.isabs(path) else os.path.join(ROOT, path)))


RUBRIC = None
_load_rubric("configs/review/rubric.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("simulate")
    s.add_argument("--key", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-raters", type=int, default=3)
    s.add_argument("--seed", type=int, default=1)
    a = sub.add_parser("analyse")
    a.add_argument("--key", required=True)
    a.add_argument("--ratings", nargs="+", required=True, help="CSV files or directories of CSVs")
    a.add_argument("--out", required=True)
    a.add_argument("--rubric", default="configs/review/rubric.json")
    a.add_argument("--reps", type=int, default=None)
    a.add_argument("--confirm-real", action="store_true", help="assert the inputs are returned expert ratings")
    a.add_argument("--keep-flagged-raters", action="store_true")
    args = ap.parse_args()
    if args.cmd == "simulate":
        fs = simulate(args.key, args.out, args.n_raters, args.seed,
                      method_effects={"D": -0.3, "E": -0.5, "G": -0.4, "B": -0.2, "I": 0.0, "C_stage": 0.3,
                                      "J": 0.2, "K": 0.3, "H": 0.4})
        print("SIMULATED ratings written (not expert data):", *fs, sep="\n  ")
    else:
        _load_rubric(args.rubric)
        res = analyse(args.key, args.ratings, args.out, args.reps, confirm_real=args.confirm_real,
                      exclude_flagged=not args.keep_flagged_raters)
        print(res["data_status"])
        print("report:", os.path.join(args.out, "agreement_report.md"))


if __name__ == "__main__":
    main()
