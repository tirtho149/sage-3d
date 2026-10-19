"""Review item B5: paired per-specimen values, leave-one-class-out, and bootstrap-vs-Holm.

For every comparison in results/statistics/comparisons.json (specimen = unit, seeds averaged):
  * the paired per-specimen values (a, b, a - b) from specimen_metrics.csv;
  * leave-one-disease-class-out: the mean difference and its sign with each class removed;
  * the smallest two-sided p the exact Wilcoxon signed-rank test can reach with the observed
    number of non-tied pairs (2 / 2^n_nonzero), and that floor after Holm at this
    comparison's rank. If the floor is above 0.05, no outcome of this design could survive
    correction, whatever the effect size; the specimen bootstrap CI, which can exclude zero
    with 6 specimens, answers a different (and at this n, anti-conservative) question.
The primary biological outcome is already fixed in reports/13 (f_overall_accept, Holm over 3).

usage: ~/Desktop/Research/.venv/bin/python experiments/review/b5_statistics.py
  -> results/review/b5_statistics/{paired_values.csv, summary.json, summary.md}
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/results/review/b5_statistics"


def main():
    os.makedirs(OUT, exist_ok=True)
    C = json.load(open(f"{ROOT}/results/statistics/comparisons.json"))
    S = pd.read_csv(f"{ROOT}/results/statistics/specimen_metrics.csv")
    comps = [c for c in C if "diff" in c]
    # Holm rank of each comparison (by raw Wilcoxon p, ascending) among those with a p
    ps = sorted([(c["wilcoxon_p"], i) for i, c in enumerate(comps)
                 if c.get("wilcoxon_p") is not None and not np.isnan(c["wilcoxon_p"])])
    m = len(ps); rank = {i: r for r, (_, i) in enumerate(ps)}
    rows, paired = [], []
    for i, c in enumerate(comps):
        a = S[S.arm == c["a"]].set_index("sid")[[c["metric"], "cls"]]
        b = S[S.arm == c["b"]].set_index("sid")[[c["metric"]]]
        j = a.join(b, lsuffix="_a", rsuffix="_b").dropna()
        j["d"] = j[f"{c['metric']}_a"] - j[f"{c['metric']}_b"]
        for sid, r in j.iterrows():
            paired.append(dict(comparison=f"{c['a']} vs {c['b']}", metric=c["metric"], specimen=sid, cls=r["cls"],
                               a=r[f"{c['metric']}_a"], b=r[f"{c['metric']}_b"], diff=r["d"]))
        nz = int((j.d.abs() > 1e-12).sum())
        floor = 2 / 2 ** nz if nz else 1.0
        holm_floor = min(1.0, floor * (m - rank[i])) if i in rank else None
        loco = {}
        for cl in sorted(j.cls.unique()):
            k = j[j.cls != cl]
            if len(k):
                loco[cl] = dict(n=int(len(k)), mean_diff=float(k.d.mean()),
                                sign_kept=bool(np.sign(k.d.mean()) == np.sign(j.d.mean())))
        rows.append(dict(comparison=f"{c['a']} vs {c['b']}", metric=c["metric"], n=int(len(j)), n_nonzero=nz,
                         n_classes=int(j.cls.nunique()), diff=float(j.d.mean()), specimen_ci=c.get("ci95_specimen_bootstrap"),
                         wilcoxon_p=c.get("wilcoxon_p"), p_holm=c.get("p_holm"), exact_p_floor=floor,
                         holm_p_floor=holm_floor, survives_possible=bool(holm_floor is not None and holm_floor <= 0.05),
                         leave_one_class_out=loco,
                         sign_robust_loco=bool(all(v["sign_kept"] for v in loco.values())) if loco else None))
    pd.DataFrame(paired).to_csv(f"{OUT}/paired_values.csv", index=False)
    json.dump(dict(n_tests_holm=m, comparisons=rows,
                   primary_biological_outcome="f_overall_accept (blinded expert rating), Holm over 3 comparisons; fixed in reports/13 before any rating"),
              open(f"{OUT}/summary.json", "w"), indent=1, default=float)
    L = ["# B5: statistics -- paired values, leave-one-class-out, and what the design can detect", "",
         f"{m} comparisons enter the Holm correction. `exact p floor` = 2/2^(non-tied pairs), the smallest two-sided p",
         "the exact Wilcoxon test can give; `Holm floor` = that floor times this comparison's Holm multiplier.",
         "If the Holm floor exceeds 0.05, no effect size could survive correction with this n.", "",
         "| comparison | metric | n (nonzero) | classes | mean diff | specimen CI | p | Holm p | exact floor | Holm floor | could survive | sign holds leaving any class out |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, p=3: ("-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{p}f}")
    for r in rows:
        ci = r["specimen_ci"]; cis = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "-"
        L.append(f"| {r['comparison']} | {r['metric']} | {r['n']} ({r['n_nonzero']}) | {r['n_classes']} | {r['diff']:+.3f} | {cis} | "
                 f"{f(r['wilcoxon_p'])} | {f(r['p_holm'])} | {f(r['exact_p_floor'])} | {f(r['holm_p_floor'])} | "
                 f"{'yes' if r['survives_possible'] else 'no'} | {'yes' if r['sign_robust_loco'] else ('-' if r['sign_robust_loco'] is None else 'NO')} |")
    L += ["", "Primary biological outcome (fixed in reports/13 before any rating): `f_overall_accept`, Holm over 3 comparisons.",
          "Per-specimen paired values: paired_values.csv."]
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
