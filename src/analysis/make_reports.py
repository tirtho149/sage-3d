"""Generate reports/05_baseline_comparison.md, 06_pilot_results.md, 07_statistical_analysis.md and NEXT_EXPERIMENT_RESULTS.md
from result files. Missing experiments are reported as NOT RUN / PENDING - never filled with estimates."""
import os, json, csv, glob
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
C = J(f"{R}/results/statistics/comparisons.json") or []; AB = J(f"{R}/results/statistics/ablation_2x2.json") or {}
SPEC = list(csv.DictReader(open(f"{R}/results/statistics/specimen_metrics.csv"))) if os.path.exists(f"{R}/results/statistics/specimen_metrics.csv") else []
ALN = J(f"{R}/results/alignment/summary.json") or []
CV = {os.path.basename(f)[15:-5]: J(f) for f in glob.glob(f"{R}/results/statistics/crossview_corr_*.json")}
cmp_ = {f"{c['a']}-{c['b']}:{c['metric']}": c for c in C}


def fmt(c):
    if not c: return "NOT RUN"
    if "diff" not in c: return f"PENDING ({c.get('note', 'insufficient pairs')}, n={c.get('n', 0)})"
    s = f"{c['mean_a']:.2f} vs {c['mean_b']:.2f}; Δ {c['diff']:+.2f} (cluster CI [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]"
    s += f"; specimen CI [{c['ci95_specimen_bootstrap'][0]:+.2f}, {c['ci95_specimen_bootstrap'][1]:+.2f}]); a>b in {c['n_positive']}/{c['n']}, n={c['n']} ({c['n_classes']} classes)"
    return s + (" ⚠ <3 classes" if c.get("ci_note") else "")


def status(c, positive_if_higher=None):
    """SUPPORTED if the specimen-bootstrap CI excludes 0 in the favourable direction (a = proposed, b = baseline)."""
    if not c or "diff" not in c: return "PENDING" if c else "NOT RUN"
    lo, hi = c["ci95_specimen_bootstrap"]
    good = (lo > 0) if c.get("higher_better", True) else (hi < 0)
    return ("SUPPORTED (pilot, small n)" if good else "NOT SUPPORTED / inconclusive") + (" — cluster CI degenerate" if c.get("ci_note") else "")


def cv_summary():
    rows = {}
    for sid, d in CV.items():
        for k, v in d.items():
            if v and v.get("median_dE_adjacent") is not None: rows.setdefault(k, []).append(v["median_dE_adjacent"])
    return {k: (float(np.median(v)), len(v)) for k, v in rows.items()}


def aln():
    if not ALN: return "NOT RUN"
    a0 = [r["A0_iou"] for r in ALN]; a3 = [r["A3s_iou"] for r in ALN]
    return f"median IoU {np.median(a0):.2f} → {np.median(a3):.2f}; improved {sum(b > a + 0.01 for a, b in zip(a0, a3))}/{len(ALN)}; held-out depth worse in {sum(r['A3s_depth_rel_err'] > r['A0_depth_rel_err'] + 0.01 for r in ALN)}/{len(ALN)}"


def main():
    cv = cv_summary()
    cvs = "; ".join(f"{k}: {m:.1f} (n={n})" for k, (m, n) in sorted(cv.items(), key=lambda x: x[1][0]))
    T = [("E1 input-view reconstruction alignment", "A0 (SAM 3D pose)", "A3* (pose-only fit)", "mask IoU (fitted) + held-out LPIPS/depth", aln(), "results/alignment/summary.json, report 01", "DONE (fit, not accuracy)"),
         ("GE diagnosis (does E change geometry?)", "D", "E", "displacement coherence", ("J vs E: " + fmt(cmp_.get("J-E:coherence"))) if cmp_.get("J-E:coherence") else "see report 02", "results/ge_diagnosis, report 02", "CONFIRMED: jitter, not deformation"),
         ("KB control of geometry", "G generic geometry (RMS-matched)", "J KB geometry", "KB geometric recall", fmt(cmp_.get("J-G:geometry_recall")), "results/statistics/comparisons.json", status(cmp_.get("J-G:geometry_recall"))),
         ("Unsupported geometry", "G", "J", "unsupported attribute rate (lower better)", fmt(cmp_.get("J-G:unsupported_rate")), "results/statistics/comparisons.json", status(cmp_.get("J-G:unsupported_rate"), positive_if_higher=False)),
         ("Texture artefacts", "E (GE)", "J", "grain ratio (lower better)", fmt(cmp_.get("J-E:grain_ratio")), "results/statistics/comparisons.json", status(cmp_.get("J-E:grain_ratio"), positive_if_higher=False)),
         ("Appearance edit without jitter", "E (GE)", "K (GE, geometry frozen)", "grain ratio / KB colour recall", fmt(cmp_.get("K-E:grain_ratio")) + " | " + fmt(cmp_.get("K-E:appearance_recall")), "comparisons.json", status(cmp_.get("K-E:grain_ratio"), positive_if_higher=False)),
         ("Joint KB geometry+appearance", "F (GE + KB, interim)", "H (J then K)", "geometric recall / colour recall", fmt(cmp_.get("H-F:geometry_recall")) + " | " + fmt(cmp_.get("H-F:appearance_recall")), "comparisons.json", status(cmp_.get("H-F:geometry_recall"))),
         ("Knowledge × geometry 2×2", "D, K, G", "H", "total KB recall; interaction (H−G)−(K−D)",
          (f"cells {json.dumps({k: (round(v, 2) if v is not None else None) for k, v in AB.get('cells', {}).items()})}; interaction {AB.get('interaction_mean')}" if AB.get("n_specimens_complete") else "PENDING (K/H runs)"), "results/statistics/ablation_2x2.json", "PENDING" if not AB.get("n_specimens_complete") else "DONE (pilot)"),
         ("Cross-view consistency (internal)", "Bmv/Cmv per-view 2D", "3D assets (E/F/K/J/G/H)", "same-point ΔE between adjacent views (lower better; D = floor)", cvs or "NOT RUN", "results/statistics/crossview_corr_*.json", "DONE (internal only)"),
         ("2D shape-capable baseline", "C-stage (mask-limited)", "I (dilated mask)", "outside-mask change / visual", "see report 05", "results/phaseA/*/I", "DONE (Peach 0087; all specimens running on CPU)"),
         ("External validity of novel views / hidden traits (H1-unseen, H4)", "—", "—", "agreement with independent real views", "NOT RUN", "—", "BLOCKED (no multi-view data; user decision)"),
         ("Biological validity", "—", "—", "blinded expert rating", "NOT RUN", "—", "BLOCKED (no raters)"),
         ("Downstream utility (§18)", "real only", "real + KB 3D renders", "macro-F1 on held-out real test", "NOT RUN", "report 10 (design + cost)", "NOT RUN (Stage-5 gate; compute)")]
    L = ["# NEXT_EXPERIMENT_RESULTS — final summary table (brief v2 §25)", "", "Generated by `src/analysis/make_reports.py` from result files. Small-n pilot: n = specimens (seeds averaged).", "",
         "| Experiment | Baseline | Proposed | Metric | Result | Evidence | Status |", "|---|---|---|---|---|---|---|"]
    L += ["| " + " | ".join(r) + " |" for r in T]
    open(f"{R}/reports/NEXT_EXPERIMENT_RESULTS.md", "w").write("\n".join(L) + "\n")
    # 07 statistical analysis
    S = ["# 07 — Statistical analysis (brief v2 §15)", "", "- **Unit:** specimen. Seeds are averaged within a specimen; views are averaged within an asset.",
         "- **Paired comparison per metric:** cluster bootstrap over disease classes (primary), specimen bootstrap (sensitivity), Wilcoxon signed-rank, Holm correction across the listed comparisons, and sign counts.",
         "- **Power caveat:** the pilot has ≤ 15 specimens in ≤ 5 classes.",
         "  - With fewer than 3 classes the cluster CI is degenerate.",
         "  - Wilcoxon cannot reach p < 0.05 with n < 6.",
         "  - The effect estimates and CIs are therefore the result; p-values are reported for completeness and none is claimed significant unless Holm p < 0.05.", "",
         "| Comparison | Metric | Result | Wilcoxon p | Holm p |", "|---|---|---|---|---|"]
    for c in C:
        S.append(f"| {c['a']} vs {c['b']} | {c['metric']} | {fmt(c)} | {c.get('wilcoxon_p', '—')} | {c.get('p_holm', '—')} |")
    S += ["", f"**2×2 ablation (geometry × KB):** `{json.dumps(AB)}`", "", "**Seed variability (J, KB geometric recall, per specimen):**"]
    seeds = {}
    for f in glob.glob(f"{R}/results/phaseA/*/J/seed*/geom_attributes.json"):
        import sys; sys.path.insert(0, f"{R}/src"); from metrics.kb_grounding import score
        sid = f.split("/")[-4]; key = "__".join(sid.split("__")[:2]); s = score(json.load(open(f)), key)["geometry_recall"]
        if s is not None: seeds.setdefault(sid, []).append(s)
    for sid, v in sorted(seeds.items()): S.append(f"- {sid}: {[round(x, 2) for x in v]} (sd {np.std(v):.2f})")
    S += ["", "**Failure rates:** see `results/statistics/qc_flags.csv` and report 09."]
    open(f"{R}/reports/07_statistical_analysis.md", "w").write("\n".join(S) + "\n")
    print("\n".join(L))


def report05():
    M2 = J(f"{R}/results/statistics/metrics2d_summary.json") or {}
    L = ["# 05 — Baseline comparison (brief v2 §7)", "", "Generated by `src/analysis/make_reports.py`. Unit = specimen (seeds averaged), then averaged over specimens.", "",
         "## What each baseline can represent (by construction)",
         "| Arm | Representation | Can change shape? | Multi-view? | KB input |", "|---|---|---|---|---|",
         "| B / C / C-NG / C-MM / C-stage | IP2P edit of the real photo, composited inside the organ mask | **no** (mask-limited) | no | generic / interim KB / KB minus geometry clauses / other disease / stage-consistent KB |",
         "| I | IP2P, mask dilated 15 % of organ diameter, image guidance 1.2 | yes (silhouette) | no | stage-consistent KB (same text as C-stage) |",
         "| Bmv / Cmv | IP2P on each of 8 ring renders independently (object-masked post hoc) | no | views edited independently | generic / KB |",
         "| D | unedited SAM 3D asset | — | yes | — |",
         "| E / F | GaussianEditor, interim hyperparameters | in principle (all params trainable), in practice jitter only | yes | generic / interim KB |",
         "| G / J | explicit geometry editor | yes (coherent) | yes | none (RMS-matched random field) / KB operators |",
         "| K / H | GaussianEditor with geometry frozen / J then K | K no; H yes (from J) | yes | KB appearance-only (fallback: stage joint) |", "",
         "## 2D arms (vs the real photo; all specimens available so far)",
         "| Arm | n specimens | ΔE inside organ | outside-mask change (× organ area) | detail ratio | KB colour recall (v2) | unsupported colour rate | background ΔE |", "|---|---|---|---|---|---|---|---|"]
    for a, v in M2.items():
        g = lambda k, d=2: "—" if v.get(k) is None else f"{v[k]:.{d}f}"
        L.append(f"| {a} | {v['n_specimens']} | {g('dE_in', 1)} | {g('outside_change')} | {g('detail_ratio')} | {g('appearance_recall')} | {g('unsupported_appearance_rate')} | {g('bg_change', 1)} |")
    L += ["", "- **Shape:** only I changes the outline (outside-mask change > 0). B/C variants are 0 by construction.",
          "- **Prompt control of colour:** KB colour recall is highest when the prompt carries the stage's colour terms (C, C-stage, I). It is lowest for the wrong-stage / no-geometry (C-NG) and mismatched (C-MM) prompts, which mostly produce unsupported colours.",
          "", "## 3D arms (vs unedited asset D, 18 fixed rig views) — see `results/statistics/specimen_metrics.csv`", ""]
    if SPEC:
        arms = ["E", "F", "G", "J", "K", "H"]; keys = ["geometry_recall", "appearance_recall", "appearance_recall_v1", "unsupported_rate", "coherence", "sil_change", "grain_ratio", "fuzz_ratio"]
        L += ["| Arm | n specimens | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
        for a in arms:
            rs = [r for r in SPEC if r["arm"] == a]
            if not rs: L.append(f"| {a} | 0 | " + " | ".join("PENDING" for _ in keys) + " |"); continue
            vals = []
            for k in keys:
                xs = [float(r[k]) for r in rs if r.get(k) not in (None, "", "None")]
                vals.append(f"{np.mean(xs):.2f}" if xs else "—")
            L.append(f"| {a} | {len(rs)} | " + " | ".join(vals) + " |")
    cv = cv_summary()
    L += ["", "## Cross-view consistency (same-point ΔE between adjacent views; lower = more consistent; internal only)", "",
          "| Arm | median over specimens | n |", "|---|---|---|"] + [f"| {k} | {m:.1f} | {n} |" for k, (m, n) in sorted(cv.items(), key=lambda x: x[1][0])]
    L += ["", "## Computational difference", "Measured medians: IP2P 2D edit about 58 s (GPU); GaussianEditor 200-step run about 2265 s; geometry edit about 44 s (incl. 18-view render). See `08_gpu_compute_audit.md`."]
    open(f"{R}/reports/05_baseline_comparison.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main(); report05()
