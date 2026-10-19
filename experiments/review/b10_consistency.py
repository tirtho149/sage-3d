"""Review item B10: one command regenerates every slide number and checks the deck against it.

Steps
  1. regenerate all macro files from results/ (slides.py -> numbers.tex, demo_figs -> demo_numbers.tex,
     edit_magnitude.py -> edit_magnitude.tex, downstream.gap_analysis -> gap_numbers.tex);
  2. compare each regenerated file with the copy the deck uses (overleaf generated/): any
     difference = a stale number on a slide;
  3. every \\Macro used in the deck's sections is defined in some generated file;
  4. the hand-typed compute table (slide "Compute") matches reports/08 (runs ok / tried).
Exit code 0 only if all checks pass.

usage: python3 experiments/review/b10_consistency.py [--no-regen]
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DECK = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "overleaf-sage-6ac80070")
SAM = os.path.expanduser("~/micromamba/envs/sam3d/bin/python")
VENV = os.path.expanduser("~/Desktop/Research/.venv/bin/python")
FILES = ["numbers.tex", "demo_numbers.tex", "edit_magnitude.tex", "gap_numbers.tex", "review_numbers.tex"]


def run(cmd, env=None):
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})})
    return p.returncode, (p.stdout + p.stderr)[-400:]


def macros(path):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", open(path).read())) if os.path.exists(path) else {}


def main():
    problems = []
    if "--no-regen" not in sys.argv:
        for cmd, env in [([SAM, "src/figures/slides.py"], {"CUDA_VISIBLE_DEVICES": ""}),
                         ([SAM, "-c", "import sys;sys.path.insert(0,'src/figures');import demo_figs as d;d.write_macros()"], {"CUDA_VISIBLE_DEVICES": ""}),
                         ([VENV, "src/analysis/edit_magnitude.py"], None),
                         ([VENV, "-m", "src.downstream.gap_analysis"], None),
                         (["python3", "experiments/review/review_macros.py"], None)]:
            rc, out = run(cmd, env)
            if rc != 0:
                problems.append(f"regeneration failed: {' '.join(cmd[1:])}: {out}")
    defined = {}
    for f in FILES:
        new, old = macros(f"{ROOT}/figures/slides/{f}"), macros(f"{DECK}/generated/{f}")
        defined.update(new)
        for k in sorted(set(new) | set(old)):
            if k != "NumGenerated" and new.get(k) != old.get(k):          # timestamp only
                problems.append(f"stale macro {f}:\\{k}: deck '{old.get(k)}' vs regenerated '{new.get(k)}'")
    used = set()
    for f in os.listdir(f"{DECK}/sections"):
        used |= set(re.findall(r"\\([A-Z][A-Za-z]+)(?![A-Za-z])", open(f"{DECK}/sections/{f}").read()))
    builtin = set(re.findall(r"\\newcommand\{\\(\w+)\}", open(f"{DECK}/preamble.tex").read()))
    sty = open(f"{DECK}/slide-template/isutrac.sty").read()
    builtin |= set(re.findall(r"\\(?:re)?newcommand\{\\(\w+)\}", sty)) | set(re.findall(r"\\def\\(\w+)", sty))
    latex = {"Large", "LARGE", "Huge", "TeX", "LaTeX", "Delta", "Rf"}
    for k in sorted(used - set(defined) - builtin - latex):
        if re.match(r"^(Ds|Tm|Mg|Pc|Cn|Cv|GR|Run|Rec|Fpr|Gap|Auc|Mag|Eone|Num|Out|Pol|Gall|St|Ctex|Warp|Abl|Surf|Curl)", k):
            problems.append(f"undefined number macro \\{k} used in the deck")
    rep = open(f"{ROOT}/reports/08_gpu_compute_audit.md").read()
    audit = {m[0]: (int(m[1]), int(m[2])) for m in re.findall(r"\| (\w[\w-]*) \| (\d+) \| (\d+) \| \d+ \|", rep)}
    deck = open(f"{DECK}/sections/act3_experiments.tex").read()
    checks = {"GaussianEditor E / F": [f"{audit['GE_E'][1]}/{audit['GE_E'][0]}", f"{audit['GE_F'][1]}/{audit['GE_F'][0]}"],
              "GaussianEditor K / H": [f"{audit['GE_K_frozen'][1] + audit['GE_H_frozen'][1]} / {audit['GE_K_frozen'][0] + audit['GE_H_frozen'][0]}"],
              "IP2P 2D edit": [f"{audit['IP2P_gpu'][1] + audit['IP2P_cpu'][1]} / {audit['IP2P_gpu'][0] + audit['IP2P_cpu'][0]}"],
              "Geometry edit J or G": [f"{audit['geometry_edit_J'][1] + audit['geometry_edit_G'][1]} / {audit['geometry_edit_J'][0] + audit['geometry_edit_G'][0]}"]}
    for row, vals in checks.items():
        line = next((ln for ln in deck.splitlines() if ln.strip().startswith(row)), "")
        for v in vals:
            if v not in line:
                problems.append(f"compute slide row '{row}' lacks '{v}' from reports/08 (line: {line.strip()[:90]})")
    if problems:
        print("INCONSISTENT:"); [print(" -", p) for p in problems]; return 1
    print(f"OK: {len(defined)} macros regenerated and identical to the deck; compute table matches reports/08.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
