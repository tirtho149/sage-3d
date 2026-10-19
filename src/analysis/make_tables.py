"""Render result files into LaTeX tables for the final report (reports/final/tables/*.tex). No hand-typed numbers."""
import json, csv, os, glob
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); T = f"{R}/reports/final/tables"; os.makedirs(T, exist_ok=True)
esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
f2 = lambda v, d=2: "--" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{d}f}" if isinstance(v, (float, int)) else esc(v))


def write(name, head, rows, cap, colspec=None):
    cs = colspec or "l" + "r" * (len(head) - 1)
    L = [r"\begin{table}[H]\centering\scriptsize", rf"\caption{{{cap}}}", r"\resizebox{\linewidth}{!}{" if len(head) > 6 else "{", rf"\begin{{tabular}}{{{cs}}}\toprule", " & ".join(head) + r" \\ \midrule"]
    L += [" & ".join(r) + r" \\" for r in rows] + [r"\bottomrule\end{tabular}}\end{table}"]
    open(f"{T}/{name}.tex", "w").write("\n".join(L) + "\n")


def alignment():
    S = json.load(open(f"{R}/results/alignment/summary.json"))
    rows = [[esc(r["specimen"].replace("__", " ")), f2(r["K_reproj_px"], 1), f2(r["A0_iou"]), f2(r["A1_iou"]), f2(r["A2_iou"]),
             f2(r["A3_iou"]) + ("$^\\dagger$" if r["A3_diverged"] else ""), r"\textbf{" + f2(r["A3s_iou"]) + "}", f2(r["A0_area_err"], 2),
             f2(r["A0_lpips"]) + r"$\to$" + f2(r["A3s_lpips"]), f2(r["A0_depth_rel_err"], 3) + r"$\to$" + f2(r["A3s_depth_rel_err"], 3), esc(r["label"])] for r in S]
    write("alignment", ["Specimen", "K px", "A0", "A1", "A2", "A3", "A3*", "area err", "LPIPS (held-out)", "depth err (held-out)", "label"], rows,
          r"E1 input-view alignment (IoU). A1 = principal point at centre; A2 = closed-form scale+translation; A3 = 7-DoF render-and-compare "
          r"(mask loss only, geometry frozen); A3* = A3 if better than A2. $^\dagger$ diverged. LPIPS and depth agreement with MoGe are never optimised.")


def specimen_metrics():
    p = f"{R}/results/statistics/specimen_metrics.csv"
    if not os.path.exists(p): return
    rows = list(csv.DictReader(open(p))); order = ["D", "E", "F", "G", "J", "K", "H"]
    g = lambda r, k: float(r[k]) if r.get(k) not in (None, "", "None") else None
    out = []
    for r in sorted(rows, key=lambda r: (r["sid"], order.index(r["arm"]) if r["arm"] in order else 99)):
        if r["arm"] not in order: continue
        out.append([esc(r["sid"].replace("__", " ")), r["arm"], r["n_seeds"], f2(g(r, "geometry_recall")), f2(g(r, "appearance_recall")),
                    f2(g(r, "unsupported_rate")), f2(g(r, "coherence")), f2(g(r, "sil_change")), f2(g(r, "grain_ratio"), 1), f2(g(r, "fuzz_ratio"), 1), f2(g(r, "dE_obj"), 1)])
    write("specimen_metrics", ["Specimen", "Arm", "seeds", "geo recall", "app recall", "unsupp. geo", "coherence", "sil.\\ change", "grain", "fuzz", "$\\Delta$E"],
          out, "Per-specimen metrics (seeds averaged). Recalls over specific detectors / scorable colour terms; -- = not applicable or not scorable.")


def comparisons():
    p = f"{R}/results/statistics/comparisons.json"
    if not os.path.exists(p): return
    C = json.load(open(p)); rows = []
    for c in C:
        if "diff" not in c: rows.append([f"{c['a']} vs {c['b']}", esc(c["metric"]), str(c["n"]), "--", "--", "--", "--", esc(c.get("note", ""))]); continue
        cs = c.get("ci95_specimen_bootstrap", [np.nan, np.nan])
        rows.append([f"{c['a']} vs {c['b']}", esc(c["metric"]), f"{c['n']} ({c['n_classes']})", f2(c["mean_a"]), f2(c["mean_b"]),
                     f"{c['diff']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]" + ("$^\\ast$" if c.get("ci_note") else ""), f"[{cs[0]:+.2f}, {cs[1]:+.2f}]",
                     f"{c.get('n_positive','-')}/{c.get('n_negative','-')}", f2(c.get("wilcoxon_p"), 3), f2(c.get("p_holm"), 3)])
    write("comparisons", ["Comparison", "Metric", "n (classes)", "mean a", "mean b", "diff [cluster CI]", "specimen CI", "a$>$b / a$<$b", "Wilcoxon p", "Holm p"], rows,
          "Specimen-level paired comparisons (seeds averaged within specimen). Cluster bootstrap over disease classes; $^\\ast$ fewer than 3 classes, so the cluster CI is degenerate or unreliable (specimen-bootstrap CI shown as sensitivity). With n $\\le$ 9 the Wilcoxon test cannot reach p $<$ 0.05 for n $<$ 6; CIs and sign counts are the primary summary.")


def ablation():
    p = f"{R}/results/statistics/ablation_2x2.json"
    if not os.path.exists(p): return
    a = json.load(open(p)); rows = [[esc(k), f2(v)] for k, v in a["cells"].items()]
    rows.append(["interaction (H$-$G)$-$(K$-$D)", f2(a.get("interaction_mean"))]); rows.append(["specimens with all four cells", str(a["n_specimens_complete"])])
    write("ablation", ["Cell", "total KB recall"], rows, "Geometry $\\times$ KB-conditioning ablation (mean over specimens with all four arms).", "lr")


def crossview():
    rows = []
    for f in sorted(glob.glob(f"{R}/results/statistics/crossview_corr_*.json")):
        d = json.load(open(f)); sid = f.split("crossview_corr_")[1][:-5]
        rows.append([esc(sid.replace("__", " "))] + [f2(d[k]["median_dE_adjacent"], 1) if k in d and d[k]["median_dE_adjacent"] is not None else "--"
                                                      for k in ("D (floor)", "Bmv", "Cmv", "E", "F", "K", "J", "G", "H")])
    if rows: write("crossview", ["Specimen", "D floor", "Bmv", "Cmv", "E", "F", "K", "J", "G", "H"], rows,
                   "Correspondence-based cross-view consistency: median CIELAB $\\Delta$E of the same surface point between adjacent ring views (lower = more consistent). Internal consistency only.")


def qc():
    p = f"{R}/results/statistics/qc_flags.csv"
    if not os.path.exists(p): return
    rows = list(csv.DictReader(open(p))); from collections import Counter
    c = Counter(); n = Counter()
    for r in rows:
        n[r["arm"]] += 1
        for fl in [x for x in r["flags"].split(";") if x]: c[(r["arm"], fl.split(":")[0].split("(")[0])] += 1
    flags = sorted({k[1] for k in c}); arms = sorted(n)
    write("qc", ["Flag"] + arms, [[esc(f)] + [f"{c[(a, f)]}/{n[a]}" for a in arms] for f in flags] or [["none"] + ["0"] * len(arms)],
          "Automated technical QC: number of flagged assets per arm (seed-level). Technical QC is not biological validation.")


if __name__ == "__main__":
    for fn in (alignment, specimen_metrics, comparisons, ablation, crossview, qc): fn()
    print(sorted(os.listdir(T)))
