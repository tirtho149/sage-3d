"""Evidence-constrained placement of local symptoms (v2), replacing uniform random placement.

Each function returns per-Gaussian weights or centres plus a `log` listing (evidence, inference, assumption,
unknown) so the report can say exactly what fixed each location and what did not.

  symptom_map_from_photo  colour of the reconstructed asset (which carries the REAL photo's colours on the visible
                          side) -> Lab score for the KB colour term (e.g. "reddish or purple cast": a* above the
                          healthy-tissue reference; "pale greenish-yellow spots": hue toward yellow and L* above
                          the lamina median). Smoothed over the surface. This is observed evidence on the visible
                          side only; on the unseen side the colours are SAM 3D's guess.
  pucker_weights          Peach Leaf Curl: puckering weighted by the reddish/purple map. The source names both
                          features for the same leaves ("Developing leaves become severely distorted (thickened and
                          puckered), and have a reddish or purple cast"); co-location at TISSUE level is an
                          inference, logged as an assumption. A map covering > 80% of the lamina is reported as
                          'effectively unconstrained'.
  gall_centres            Corn Smut: centres on kernel tissue only (UNL: galls "develop on the ear replacing
                          kernels"); along-ear position uniform over kernel tissue (no source gives it); diameters
                          log-uniform within the SAGE range (1-30 cm) capped at the ear diameter, converted to asset
                          units with an ASSUMED ear length (default 18 cm; no ruler in the photo).
  spot_map                Tomato Leaf Mold: upper-surface spots from the photo colours (pale greenish-yellow);
                          the lower-surface mold is placed directly beneath them (KB: "below leaf spots").
"""
import numpy as np
from scipy.spatial import cKDTree
from .anatomy import rgb, srgb_to_lab


def smooth_on_surface(X, f, k=32, iters=4):
    _, nb = cKDTree(X).query(X, k=k)
    for _ in range(iters):
        f = f[nb].mean(1)
    return f


def lab_of(v):
    return srgb_to_lab(rgb(v))


def reddish_map(F, thr_sigma=1.0):
    """a* (green-red axis) relative to the lamina: z-score of a* against the greener half of the lamina."""
    lab = lab_of(F["v"]); a = lab[:, 1]; ref = a[a <= np.median(a)]
    z = (a - ref.mean()) / (ref.std() + 1e-6); m = smooth_on_surface(F["X"], (z > 2.0 * thr_sigma).astype(float))
    return np.clip(m, 0, 1), dict(feature="a* z-score > 2 vs greener half of lamina", frac=float((m >= 0.5).mean()))


def pucker_weights(F):
    m, info = reddish_map(F); frac = info["frac"]
    log = [dict(kind="evidence", text="KB: puckering and reddish/purple cast co-occur on developing leaves (leaf level)"),
           dict(kind="assumption", text="tissue-level co-location of puckering with visible reddish/purple tissue"),
           dict(kind="measured", text=f"reddish/purple map covers {frac:.2f} of the asset")]
    if frac > 0.8 or frac < 0.02:
        log.append(dict(kind="unknown", text="map is (nearly) uniform: placement effectively unconstrained; weights set to 1"))
        return np.ones(len(m)), log, frac
    w = 0.25 + 0.75 * m                     # never exactly zero: the unseen side has no colour evidence
    return w, log, frac


def gall_centres(E, n=3, ear_len_cm=18.0, rng=None, min_sep=1.0, hidden_only=False):
    """Returns centres (k,3), outward normals, diameters (asset units), and the log.
    hidden_only=True is the observed-evidence 'preserve' mode (evidence_policy.py): new galls may
    only start on kernel tissue NOT visible from the input camera, so they cannot contradict the photo."""
    rng = rng or np.random.default_rng(0); X = E["X"]; a = E["a"]; ear_len = float(np.ptp(a)); cm = ear_len / ear_len_cm
    rad = np.linalg.norm((X - X.mean(0)) - a[:, None] * E["axis"], axis=1); ear_d_cm = 2 * np.percentile(rad, 90) / cm
    cand = np.nonzero(E["kernel"] & ~E["gall"] & (~E["visible"] if hidden_only else True))[0]
    dmin, dmax = 1.0, min(30.0, max(1.5, ear_d_cm))
    C, N, D = [], [], []
    for _ in range(200):
        if len(C) == n or len(cand) == 0: break
        i = cand[rng.integers(len(cand))]; d = float(np.exp(rng.uniform(np.log(dmin), np.log(dmax))))
        if any(np.linalg.norm(X[i] - c) < min_sep * (d + dd) / 2 * cm for c, dd in zip(C, D)): continue
        C.append(X[i]); N.append(E["n"][i]); D.append(d)
    log = [dict(kind="evidence", text="UNL CropWatch: galls 'develop on the ear replacing kernels' -> centres on kernel tissue only"),
           dict(kind="evidence", text="SAGE KB: diameters 1-30 cm -> log-uniform in [1, min(30, ear diameter)] cm"),
           dict(kind="assumption", text=f"ear length {ear_len_cm} cm (no scale in the photo) -> 1 cm = {cm:.4f} asset units; ear diameter {ear_d_cm:.1f} cm"),
           dict(kind="unknown", text="position along the ear (tip vs base) not given by any source: uniform over kernel tissue"),
           dict(kind="assumption", text=f"number of new galls = {n} (no source gives a count)")]
    if hidden_only:
        log.append(dict(kind="policy", text="preserve mode: centres restricted to kernel tissue not visible in the input photo"))
    return np.array(C), np.array(N), np.array(D) * cm, dict(cm_per_unit=1 / cm, diam_cm=D, ear_diameter_cm=ear_d_cm, candidates=int(len(cand))), log


def spot_map(F):
    """Pale greenish-yellow spots on the visible (upper) side: hue toward yellow and lighter than the lamina."""
    lab = lab_of(F["v"]); L, A, B = lab.T; hue = np.degrees(np.arctan2(B, A))
    lam = (B > 5)                                                       # leaf tissue (exclude dark background bleed)
    z = lambda x: (x - np.median(x[lam])) / (x[lam].std() + 1e-6)
    s = (z(L) > 0.5) & (hue < np.median(hue[lam]) - 3) & lam
    m = smooth_on_surface(F["X"], s.astype(float), k=24, iters=2)
    log = [dict(kind="evidence", text="KB: 'Pale greenish-yellow spots ... form on the upper sides of leaves' -> spots observed in the photo"),
           dict(kind="measured", text=f"spot map covers {float((m >= 0.5).mean()):.3f} of the asset (L* z > 0.5 and hue toward yellow)"),
           dict(kind="evidence", text="KB: mold 'forms on the lower leaf surface below leaf spots' -> mold footprint = spot footprint, lower layer")]
    return np.clip(m, 0, 1), log
