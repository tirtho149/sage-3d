"""Directional curl / roll / cup operator for a leaf asset (v2), driven by the KB phrase.

Target (exact for a flat sheet; regularised by ARAP for a curved one): an isometric bend about the midrib.
For a Gaussian at lateral geodesic distance a = |lat| from the midrib and half-width w(u):
   onset  a0 = r0 * w             (r0 = 0: whole-width roll / cup; r0 = 0.5: "at the edges")
   arc    s  = max(a - a0, 0),     curvature kappa = theta_margin / (w (1 - r0))  (every margin turns by theta_margin)
   target displacement  = (sin(kappa s)/kappa - s) t_lat + sgn (1 - cos(kappa s))/kappa n_adaxial
   sgn = +1: margins move toward the ADAXIAL side (upward, involute); sgn = -1: toward the abaxial side (downward,
   revolute, "rolled under"). Lengths along the bend are preserved by construction (isometry), so tissue does not
   stretch; the ARAP term in deform.py keeps small patches rigid, which also preserves existing relief.

Direction comes ONLY from the KB text (DIRECTION lexicon below). If the phrase names curl/roll/cup without a
direction, the operator ABSTAINS (no direction is invented); a caller may still run explicit, labelled hypotheses.
Magnitude (theta_margin) is NOT given by any source: lexicon defaults below, reported as an assumption.
"""
import os, sys, re, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src")
DIRECTION = [(r"\b(roll\w*|curl\w*|cupp?\w*)\s+(down\w*|under\w*)|\b(down\w*|under)\b[^.]{0,20}\b(curl|roll|cup)|rolled under", -1),
             (r"\b(roll\w*|curl\w*|cupp?\w*)\s+(up\w*|inward\w*)|\bup\w*\b[^.]{0,20}\b(curl|roll|cup)", +1)]
EXTENT = [(r"\bedges?\b|\bmargins?\b", 0.5)]            # "at the edges", "margins rolled": bend the outer half only
MAGNITUDE = {"cup": 45.0, "curl": 90.0, "roll": 150.0}  # degrees at the margin (assumption, not from a source)


def parse(phrase):
    p = phrase.lower(); kind = "roll" if re.search(r"\broll", p) else ("cup" if re.search(r"\bcup", p) else ("curl" if re.search(r"\bcurl", p) else None))
    sgn = next((s for pat, s in DIRECTION if re.search(pat, p)), None); r0 = next((v for pat, v in EXTENT if re.search(pat, p)), 0.0)
    return dict(kind=kind, direction=sgn, r0=r0, theta_deg=MAGNITUDE.get(kind), phrase=phrase,
                status="ok" if (kind and sgn) else ("abstain: no direction in the source phrase" if kind else "abstain: no curl/roll/cup term"))


def _smooth(X, f, k=48, iters=6):
    from scipy.spatial import cKDTree
    _, nb = cKDTree(X).query(X, k=k)
    for _ in range(iters):
        f = f[nb].mean(1)
    return f


def curl_target(F, direction, theta_deg, r0=0.0):
    """The frames (lateral tangent, adaxial normal) are smoothed over the surface (48-NN, 6 passes): per-Gaussian PCA
    frames of a SAM 3D sheet are noisy. The target itself is not smoothed (tests/test_v2_operators.py: smoothing the
    bend stretched a flat sheet by 12% at p95)."""
    X = F["X"]; n = _smooth(X, F["n"]); n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    t = F["t_lat"] * np.sign(F["lat"] + 1e-12)[:, None]            # one consistent lateral direction (t_+) ...
    t = _smooth(X, t); t -= (t * n).sum(1, keepdims=True) * n; t /= np.linalg.norm(t, axis=1, keepdims=True).clip(1e-12)
    t *= np.sign(F["lat"] + 1e-12)[:, None]                        # ... then pointing away from the midrib again
    a = np.abs(F["lat"]); w = F["half"]; s = np.maximum(a - r0 * w, 0.0); kappa = np.radians(theta_deg) / np.maximum(w * (1 - r0), 1e-9)
    ks = kappa * s; dt = np.where(ks > 1e-9, np.sin(ks) / kappa - s, 0.0); dn = np.where(ks > 1e-9, (1 - np.cos(ks)) / kappa, 0.0)
    return dt[:, None] * t + direction * dn[:, None] * n          # no smoothing of the target itself: averaging a bend shortens the arc


def run_curl(F, direction, theta_deg, r0, out_dir, label, seed=0, iters=400):
    """Writes out_dir/asset.ply (+ .json from deform.run) and returns the deform report."""
    from geometry_editor.deform import run as deform
    os.makedirs(out_dir, exist_ok=True); tgt = curl_target(F, direction, theta_deg, r0)
    np.save(f"{out_dir}/curl_target.npy", tgt.astype(np.float32)); np.save(f"{out_dir}/normals.npy", F["n"].astype(np.float32))
    spec = dict(seed=seed, iters=iters, mode="sheet", normals_npy=f"{out_dir}/normals.npy",
                ops=[dict(op="vector", target_npy=f"{out_dir}/curl_target.npy", kb_term=label)],
                lambdas=dict(edit=1.0, arap=1.0, smooth=0.01, anchor=0.0), arap_coarse=20.0)
    return deform(F["ply"], spec, f"{out_dir}/asset.ply")
