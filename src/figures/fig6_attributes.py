"""Figure 6 - Requested (KB, selected stage) vs realised attributes, per class. Cell = fraction of (specimen, seed) assets/images
in which the attribute is detected. Starred columns are required by the KB stage; detections in unstarred columns are unsupported.
3D arms: geometric detectors (specific only) + colour detectors on rig renders; 2D arms: colour detectors in the organ mask."""
import os, sys, json, glob
import numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures.common import ROOT, TW, save
from metrics.kb_grounding import score as kb_geo, OP2ATTR
from metrics.appearance_kb import main2d, required as app_required

GEO = ["distorted", "puckered", "thickened", "outgrowth"]; APP = ["redder", "yellower_or_paler", "darker", "greyer"]
ARMS3D = ["E", "F", "G", "J", "K", "H"]; ARMS2D = ["B", "C", "C_nogeo", "C_mismatch", "C_stage", "I"]
KEY = lambda sid: "Citrus__Canker__leaf" if sid == "Citrus__Canker__0004" else "__".join(sid.split("__")[:2])


def collect():
    specs = json.load(open(f"{ROOT}/data/kb/structured_specs.json")); out = {}
    for P in sorted(glob.glob(f"{ROOT}/results/phaseA/*__*__*")):
        sid = os.path.basename(P); key = KEY(sid); cls = "__".join(sid.split("__")[:2]); D = out.setdefault(cls, {})
        for arm in ARMS3D:
            for d in sorted(glob.glob(f"{P}/{arm}/seed*")):
                if not os.path.isdir(d): continue
                det = set()
                if os.path.exists(f"{d}/geom_attributes.json"): det |= set(kb_geo(json.load(open(f"{d}/geom_attributes.json")), key)["detected"])
                if os.path.exists(f"{d}/appearance_kb.json"): det |= set(json.load(open(f"{d}/appearance_kb.json"))["detected"])
                if os.path.exists(f"{d}/geom_attributes.json") or os.path.exists(f"{d}/appearance_kb.json"): D.setdefault(arm, []).append(det)
        I = f"{ROOT}/data/inputs/{sid}"
        for arm in ARMS2D:
            for f in sorted(glob.glob(f"{ROOT}/results/pilot/{sid}/{arm}/seed*.png") + glob.glob(f"{P}/{arm}/seed*.png")):
                D.setdefault(arm, []).append(set(main2d(f"{I}/image.png", f"{I}/0.png", f, key)["detected"]))
        req = {OP2ATTR[g["op"]] for g in specs[key]["geometry"]["operators"]} | app_required(key)[0]
        D["_req"] = sorted(set(D.get("_req", [])) | req)
    return out


def main():
    C = collect(); classes = [c for c in C if any(k != "_req" for k in C[c])]
    fig, axs = plt.subplots(1, len(classes), figsize=(TW, 2.3), squeeze=False)
    cols = GEO + APP
    for ax, cls in zip(axs[0], classes):
        D = C[cls]; arms = [a for a in ARMS2D + ARMS3D if a in D]
        M = np.array([[np.mean([c in s for s in D[a]]) for c in cols] for a in arms])
        ax.imshow(M, vmin=0, vmax=1, cmap="Greens", aspect="auto")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([("*" if c in D["_req"] else "") + c.replace("yellower_or_paler", "yellow/pale") for c in cols], rotation=60, ha="right", fontsize=4.3)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels([f"{a} (n={len(D[a])})" for a in arms], fontsize=4.3)
        ax.set_title(cls.replace("__", " ").replace("_", " "), fontsize=5)
        for i in range(len(arms)):
            for j in range(len(cols)):
                ax.text(j, i, f"{M[i, j]:.1f}" if M[i, j] > 0 else "", ha="center", va="center", fontsize=3.5, color="black" if M[i, j] < 0.6 else "white")
        ax.axvline(len(GEO) - 0.5, color="#888", lw=0.4)
    fig.tight_layout()
    save(fig, "fig6_attributes", ["results/phaseA/*/*/seed*/{geom_attributes,appearance_kb}.json", "results/pilot/*/*/seed*.png"],
         "Requested vs realised KB attributes. * = required by the selected KB stage; detections in unstarred columns are unsupported. "
         "Geometric detectors: specific only (depression excluded). Colour detectors: CIELAB shifts >= 5 units. Detection is not biological validation.", vector=True)
    json.dump({c: {a: [sorted(s) for s in v] if a != "_req" else v for a, v in d.items()} for c, d in C.items()}, open(f"{ROOT}/results/statistics/attributes_detected.json", "w"), indent=1)


if __name__ == "__main__":
    main()
