"""Appearance KB-grounding and cross-view consistency of an edit (brief s12-s14).

Per view (pixels covered in both D and X): mean dL*, da*, db*, dChroma (CIELAB, X - D).
Term detectors (threshold T = 5 Lab units ~ 2 JND, fixed a priori; never tuned on outputs):
  red/reddish                   da* >= +T
  purple                        db* <= -T and da* >= 0      [added post hoc: the first version mapped "purple" to da* only,
                                which cannot see a red leaf turning purple (blue shift); deviation logged in reports/04.
                                A phrase naming several colours ("reddish or purple") is satisfied by ANY of them (OR-group).]
  yellow/greenish-yellow/pale   db* >= +T  or  (dL* >= +T and dChroma <= 0)   [pale = lighter AND not more saturated;
                                the first version used dL* >= +T alone, which counted saturated purple lightening as "pale" -
                                corrected after seeing Peach 0087 2D outputs; deviation logged in reports/04]
  brown/black/dark/sooty        dL* <= -T
  gray/grey/powdery             dChroma <= -T
  lower-surface localisation    dE(bottom view) / dE(top view) >= 1.5   (only if the KB phrase names the lower surface)
Terms with no detector (e.g. 'velvety', 'oily', 'halo', 'spots' shape) are reported as not scorable.
Cross-view consistency (internal only): per-view mean (dL, da, db) vectors; consistency = 1 - mean_view ||v - v_bar|| / ||v_bar||.
usage: appearance_kb.py <D_render> <X_render> <rig.json> <class_key> [out.json]   (or --2d <input.png> <mask.png> <edit.png> <class_key>)
"""
import os, sys, json, re
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{R}/src")
from rendering.rig_render import load_view
T = 5.0
TERMS = [(r"\b(red\w*)\b", "redder"), (r"\b(purpl\w*)\b", "purpler"), (r"\b(yellow\w*|pale)\b", "yellower_or_paler"), (r"\b(brown\w*|black\w*|dark\w*|sooty)\b", "darker"),
         (r"\b(gr[ae]y|powdery)\b", "greyer")]


def lab_delta(a, b, m):
    A, B = rgb2lab(a / 255.0)[m], rgb2lab(b / 255.0)[m]; d = B - A
    ch = np.hypot(B[:, 1], B[:, 2]) - np.hypot(A[:, 1], A[:, 2])
    return dict(dL=float(d[:, 0].mean()), da=float(d[:, 1].mean()), db=float(d[:, 2].mean()), dC=float(ch.mean()), dE=float(np.linalg.norm(d, axis=1).mean()))


def detect(s):
    return dict(redder=s["da"] >= T, purpler=(s["db"] <= -T) and (s["da"] >= 0), yellower_or_paler=(s["db"] >= T) or (s["dL"] >= T and s["dC"] <= 0), darker=s["dL"] <= -T, greyer=s["dC"] <= -T)


def required_groups(key):
    """One OR-group per KB appearance phrase (e.g. 'reddish or purple cast' -> {redder, purpler})."""
    S = json.load(open(f"{R}/data/kb/structured_specs.json"))[key]; groups = []
    for a in S["appearance"]:
        g = {name for pat, name in TERMS if re.search(pat, a["phrase"], re.I)}
        if g: groups.append(g)
    return groups


def required(key):
    S = json.load(open(f"{R}/data/kb/structured_specs.json"))[key]; req, lower, unscorable = set(), False, []
    for a in S["appearance"]:
        hit = False
        for pat, name in TERMS:
            if re.search(pat, a["phrase"], re.I): req.add(name); hit = True
        if re.search(r"lower (leaf )?surface|under\s?side|beneath", a["phrase"], re.I): lower = True; hit = True
        if not hit: unscorable.append(a["phrase"])
    return req, lower, unscorable


def score(per_view, key, top=None, bottom=None):
    allv = {k: float(np.mean([v[k] for v in per_view])) for k in ("dL", "da", "db", "dC", "dE")}
    det = {k for k, v in detect(allv).items() if v}; req, lower, uns = required(key); groups = required_groups(key)
    out = dict(mean=allv, detected=sorted(det), required=sorted(req), required_groups=[sorted(g) for g in groups], unscorable_phrases=uns,
               appearance_recall=(sum(bool(det & g) for g in groups) / len(groups)) if groups else None,
               appearance_recall_v1_redder_only=(("redder" in det) * 1.0 if any("redder" in g for g in groups) else None),
               unsupported_appearance_rate=(len(det - req) / len(det)) if det else 0.0)
    if lower and top is not None and bottom is not None:
        out["lower_surface_ratio"] = bottom["dE"] / max(top["dE"], 1e-9); out["lower_surface_localised"] = out["lower_surface_ratio"] >= 1.5
    if len(per_view) > 1:
        V = np.array([[v["dL"], v["da"], v["db"]] for v in per_view]); vb = V.mean(0)
        out["crossview_consistency"] = float(1 - np.linalg.norm(V - vb, axis=1).mean() / max(np.linalg.norm(vb), 1e-9))
    return out


def main3d(Ddir, Xdir, rig, key):
    pv, by = [], {}
    for v in rig["views"]:
        i = v["id"]; a, aa, _ = load_view(Ddir, i); b, ab, _ = load_view(Xdir, i); a, b = a.astype(float), b.astype(float)
        m = (aa > 0.5) & (ab > 0.5)
        if m.sum() < 50: continue
        s = lab_delta(a, b, m); pv.append(s); by[(v["az"], v["el"])] = s
    top = by.get((0.0, 80.0)); bot = by.get((0.0, -80.0))
    return dict(score(pv, key, top, bot), per_view=pv)


def main2d(inp, mask, ed, key):
    a = np.asarray(Image.open(inp).convert("RGB"), float); b = np.asarray(Image.open(ed).convert("RGB").resize(a.shape[1::-1]), float)
    m = np.asarray(Image.open(mask).convert("L")) > 127
    return score([lab_delta(a, b, m)], key)


if __name__ == "__main__":
    if sys.argv[1] == "--2d":
        r = main2d(*sys.argv[2:6])
    else:
        r = main3d(sys.argv[1], sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4])
        if len(sys.argv) > 5: json.dump(r, open(sys.argv[5], "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k != "per_view"}))
