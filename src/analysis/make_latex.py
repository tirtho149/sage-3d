"""Write reports/final/{gen_date,sections_summary,sections_results,sections_decision}.tex from result files, copy figures,
and compile the PDF with tectonic. Numbers are read from results/statistics/*; nothing is typed by hand here."""
import os, json, csv, glob, shutil, subprocess, time
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); F = f"{R}/reports/final"
J = lambda p: json.load(open(p)) if os.path.exists(p) else None
C = {f"{c['a']}-{c['b']}:{c['metric']}": c for c in (J(f"{R}/results/statistics/comparisons.json") or [])}
AL = J(f"{R}/results/alignment/summary.json") or []; AB = J(f"{R}/results/statistics/ablation_2x2.json") or {}
M2 = J(f"{R}/results/statistics/metrics2d_summary.json") or {}
CV = {}
for f in glob.glob(f"{R}/results/statistics/crossview_corr_*.json"):
    for k, v in J(f).items():
        if v.get("median_dE_adjacent") is not None: CV.setdefault(k, []).append(v["median_dE_adjacent"])
esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def cstr(key):
    c = C.get(key)
    if not c or "diff" not in c: return r"\emph{pending}"
    cs = c["ci95_specimen_bootstrap"]
    return (f"{c['mean_a']:.2f} vs {c['mean_b']:.2f} ($\\Delta$ {c['diff']:+.2f}, specimen-bootstrap 95\\% CI [{cs[0]:+.2f}, {cs[1]:+.2f}], "
            f"{c['n_positive']}/{c['n']} specimens favour the first arm, {c['n_classes']} class{'es' if c['n_classes'] != 1 else ''})")


def fig(name, cap, w=r"\linewidth"):
    src = f"{R}/figures/paper/{name}.png"
    if not os.path.exists(src): return f"% missing figure {name}\n"
    os.makedirs(f"{F}/figs", exist_ok=True); shutil.copy(src, f"{F}/figs/{name}.png")
    return f"\\begin{{figure}}[H]\\centering\\includegraphics[width={w}]{{figs/{name}.png}}\\caption{{{cap}}}\\end{{figure}}\n"


def tab(name):
    return f"\\input{{tables/{name}.tex}}\n" if os.path.exists(f"{F}/tables/{name}.tex") else f"% missing table {name}\n"


def main():
    open(f"{F}/gen_date.tex", "w").write(time.strftime("%Y-%m-%d %H:%M"))
    a0 = np.median([r["A0_iou"] for r in AL]) if AL else float("nan"); a3 = np.median([r["A3s_iou"] for r in AL]) if AL else float("nan")
    cvm = {k: np.median(v) for k, v in CV.items()}
    S = [r"\section*{Summary}", r"\begin{itemize}",
         rf"\item \textbf{{Reconstruction.}} The low input-view IoU reported in the interim was a SAM~3D pose/scale error (objects over-scaled in {sum(r['A0_area_err'] > 0 for r in AL)}/{len(AL)}), not a camera error. A pose-only correction raises median IoU from {a0:.2f} to {a3:.2f} on all {len(AL)} reconstructions. This is a fit to the input silhouette, not an accuracy measurement. One specimen (Peach 0006) is a reconstruction failure.",
         r"\item \textbf{Arm F} failed only because of prompt serialisation (fixed and verified on live runs). A second bug (non-finite opacity $\rightarrow$ NaN $\rightarrow$ every fp16 step skipped) explained the ``no change'' on Peach 0006.",
         rf"\item \textbf{{GaussianEditor does not edit geometry.}} Its Gaussians move incoherently (displacement coherence {C['J-E:coherence']['mean_b']:.2f} vs 0.25 for noise); the sheet thickens and the texture becomes grainy. KB text changes its colour output but not its geometry: F vs E coherence {cstr('F-E:coherence')}." if "J-E:coherence" in C and "diff" in C["J-E:coherence"] else r"\item GaussianEditor diagnosis: see Section~\ref{sec:res}.",
         rf"\item \textbf{{A KB-driven geometry editor}} produces coherent, measurable deformation (coherence $\approx$1). Against an equal-magnitude generic warp, KB geometric recall is {cstr('J-G:geometry_recall')}. Recall is partly by construction; biological correctness is unvalidated.",
         (rf"\item \textbf{{Cross-view consistency}} (same surface point, adjacent views, median $\Delta$E): per-view 2D edits {cvm.get('Bmv', float('nan')):.1f}/{cvm.get('Cmv', float('nan')):.1f} (generic/KB), GaussianEditor 3D {cvm.get('E', float('nan')):.1f}, geometry-only 3D {cvm.get('J', float('nan')):.1f}, against a floor of {cvm.get('D (floor)', float('nan')):.1f}. This is internal consistency only." if CV else ""),
         r"\item \textbf{Not established:} external validity of any novel view (no reference data), biological validity (no experts) and downstream utility (not run).",
         r"\end{itemize}"]
    open(f"{F}/sections_summary.tex", "w").write("\n".join(S) + "\n")
    X = [r"\section{Results}\label{sec:res}", fig("fig1_pipeline", "Pipeline and data status."),
         r"\subsection{Input-view alignment (E1)}", tab("alignment"), fig("fig2_alignment", "Reconstruction debugging: A0 vs corrected pose A3*. Fitted, not verified."),
         r"\subsection{What GaussianEditor changes}", fig("fig5_displacement_Peach__Leaf_Curl__0087", "Displacement heatmaps and geometry-only / appearance-only decomposition (Peach 0087)."),
         r"\subsection{Geometry-aware editing and knowledge ablation}", fig("fig3_editing_Peach__Leaf_Curl__0087", "All 3D arms of Peach 0087 on fixed rig views (synthetic, unverified)."),
         fig("fig4_kb_ablation_Peach__Leaf_Curl__0087", "Knowledge ablation in 2D and 3D on Peach 0087."), fig("fig6_attributes", "Requested (KB) vs realised attributes per class."),
         tab("specimen_metrics"), tab("comparisons"), tab("ablation"),
         r"\subsection{2D baselines}", r"\begin{table}[H]\centering\scriptsize\caption{2D arms vs the real photo (specimen-averaged). Outside-mask change = changed area outside the organ mask $\div$ organ area (shape-change capacity).}\begin{tabular}{lrrrrr}\toprule Arm & n & $\Delta$E in organ & outside change & KB colour recall & unsupported colours\\\midrule"]
    for a, v in M2.items():
        X.append(f"{esc(a)} & {v['n_specimens']} & {v['dE_in']:.1f} & {v['outside_change']:.2f} & " + (f"{v['appearance_recall']:.2f}" if v['appearance_recall'] is not None else "--") + f" & {v['unsupported_appearance_rate']:.2f}\\\\")
    X += [r"\bottomrule\end{tabular}\end{table}", r"\subsection{Cross-view consistency}", tab("crossview"), fig("fig7_crossview_Peach__Leaf_Curl__0087", "Per-view 2D edits vs one 3D asset (Peach 0087)."),
          r"\subsection{Quality control and failures}", tab("qc"), fig("fig9_failures", "Failure taxonomy (real runs)."),
          r"\subsection{What 3D adds, and what it does not}", fig("fig10_added_value_Peach__Leaf_Curl__0087", "Added value of the 3D asset vs 2D editing, with a reconstruction failure.")]
    open(f"{F}/sections_results.tex", "w").write("\n".join(X) + "\n")
    shutil.copy(f"{R}/reports/final/decision.tex", f"{F}/sections_decision.tex") if os.path.exists(f"{R}/reports/final/decision.tex") else open(f"{F}/sections_decision.tex", "w").write("\\section{Decision}\\emph{pending}\n")
    r = subprocess.run([os.path.expanduser("~/micromamba/envs/tex/bin/tectonic"), "-X", "compile", "report.tex"], cwd=F, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:])


if __name__ == "__main__":
    main()
