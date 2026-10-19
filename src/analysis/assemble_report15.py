"""Fill reports/15_revised_scientific_report.template.md with numbers from result files -> reports/15_revised_scientific_report.md."""
import os, re, json, glob
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T = open(f"{R}/reports/15_revised_scientific_report.template.md").read(); md = open(f"{R}/results/demos/summary.md").read()


def section(title):
    m = re.search(r"## " + re.escape(title) + r"[^\n]*\n(.*?)(?=\n## |\Z)", md, re.S); return m.group(1).strip() if m else "(missing)"


comp = json.load(open(f"{R}/results/statistics/comparisons.json"))
rows = ["| comparison | n (classes) | means a / b | diff [95% specimen CI] | Holm p |", "|---|---|---|---|---|"]
for c in comp:
    if c.get("p_holm") is None or c["metric"].endswith("_v1"): continue
    lo, hi = c["ci95_specimen_bootstrap"]
    rows.append(f"| {c['a']} vs {c['b']} {c['metric']} | {c['n']} ({c['n_classes']}) | {c['mean_a']:.3f} / {c['mean_b']:.3f} | {c['diff']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {c['p_holm']:.3f} |")
ab = json.load(open(f"{R}/results/statistics/ablation_2x2.json"))
ds = json.load(open(f"{R}/results/downstream_minimal/summary.json"))
drows = ["| k | KB-3D minus real only | KB-3D minus unedited 3D renders |", "|---|---|---|"]
dd = {(e["b"], e["k"]): e for e in ds["families"]["decision"] if e["a"] == "KB3D" and e["metric"] == "macro_f1"}
for k in ds["ks"]:
    f = lambda e: f"{e['est']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}] (Holm p {e['p_holm']:.2f})" if e else "-"
    drows.append(f"| {k} | {f(dd.get(('R', k)))} | {f(dd.get(('D', k)))} |")
down = ("Within-crop disease classification on 2,936 held-out real SAGE photos (5 crops), frozen DINOv2 + linear probe, "
        f"{ds['n_draws']} paired draws per k; pooled macro-F1:\n\n" + "\n".join(drows) +
        f"\n\n**Pre-registered verdict: {ds['decision']['verdict']}.** Rule: {ds['decision']['rule']}. The only reliable gain came from ordinary "
        "2D augmentation of the specimens' own photos; renders lowered real-photo recall, so the render-to-photo gap precedes any knowledge effect.")
orient, stress, galls = [], [], []
for p in glob.glob(f"{R}/results/demos/*/*/metrics.json"):
    d = json.load(open(p))
    for a in d.get("assumptions", []):
        m = re.match(r"([0-9.]+) of input-visible", a.get("check", "")); orient += [float(m.group(1))] if m else []
        m = re.search(r"Isomap stress ([0-9.]+)", a.get("check", "")); stress += [float(m.group(1))] if m else []
    for g in d.get("real_galls_in_photo") or []:
        if d.get("scale"): galls.append(max(g["extent_cm"]))
fmt = lambda v, d=2: f"{min(v):.{d}f}-{max(v):.{d}f}" if v else "n/a"
rep = {"STATS_TABLE": "\n".join(rows), "AB_N": str(ab["n_specimens_complete"]), "AB_INT": f"{ab['interaction_mean']:+.2f}",
       "MANGO_TABLE": section("Mango leaves + 'Leaves curl upwards' (direction control)"), "PEACH_TABLE": section("Peach Leaf Curl"),
       "CORN_TABLE": section("Corn Smut"), "TOMATO_TABLE": section("Tomato Leaf Mold: where the mold-coloured pixels land"), "DOWNSTREAM": down,
       "ORIENT": fmt(orient), "STRESS": fmt(stress, 3), "REAL_GALLS": fmt(galls, 1) + f" (n = {len(galls)} components)"}
for k, v in rep.items(): T = T.replace("{{" + k + "}}", v)
assert "{{" not in T, re.findall(r"\{\{\w+\}\}", T)
open(f"{R}/reports/15_revised_scientific_report.md", "w").write(T); print("written", len(T))
