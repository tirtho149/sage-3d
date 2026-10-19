"""Anatomically separate upper (adaxial) and lower (abaxial) leaf surfaces for a single-sheet SAM 3D leaf (v2).

Problem: SAM 3D stores one view-independent colour per Gaussian and one sheet, so the underside of every leaf is
the upper surface seen from behind (the unedited Tomato asset shows the photo's upper-surface spots from below).
A symptom that the KB places on one surface (e.g. Tomato Leaf Mold: mold "on the lower leaf surface below leaf
spots") cannot be represented without appearing on the other surface too.

Construction:
  * every Gaussian i is replaced by two thin Gaussians at mu_i + delta n_i (upper) and mu_i - delta n_i (lower),
    n = smoothed adaxial normal (anatomy.adaxial_normals); each is re-oriented so its thin axis is n, keeping the
    two largest of its original scales in the tangent plane; thickness scale = delta / 3.
    delta = 0.002 L by default (leaf half-thickness; a 10 cm leaf -> 0.2 mm; an assumption, reported).
  * upper colour = original colour (on the visible side: the REAL photo's colours).
  * lower BASE colour = original colour with upper-surface lesions removed (lesion Gaussians take the mean colour of
    the 16 nearest non-lesion Gaussians). The true abaxial colour is unknown (no photo of the underside): labelled
    SYNTHETIC, unverified.
  * lower symptoms (e.g. mold) are written to the lower layer only, at the KB-given footprint.
Layer ids are stored in <out>.layers.npy (0 = upper, 1 = lower) for the surface-leakage metrics.
"""
import json
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
from .anatomy import rgb, rgb_to_dc


def lab_to_srgb(lab):
    L, a, b = lab.T; fy = (L + 16) / 116; fx = fy + a / 500; fz = fy - b / 200
    f = np.stack([fx, fy, fz], 1); X = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    M = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]); l = X @ M.T
    return np.clip(np.where(l > 0.0031308, 1.055 * np.abs(l) ** (1 / 2.4) - 0.055, 12.92 * l), 0, 1)


def quat_to_mat(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1), np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                     np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], 1)


def mat_to_quat(R):
    t = np.trace(R, axis1=1, axis2=2); w = np.sqrt(np.clip(1 + t, 1e-12, None)) / 2
    q = np.stack([w, (R[:, 2, 1] - R[:, 1, 2]) / (4 * w), (R[:, 0, 2] - R[:, 2, 0]) / (4 * w), (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)], 1)
    bad = w < 1e-3                                         # fall back to a robust branch near 180 degrees
    if bad.any():
        from scipy.spatial.transform import Rotation
        xyzw = Rotation.from_matrix(R[bad]).as_quat(); q[bad] = xyzw[:, [3, 0, 1, 2]]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def smooth_normals(X, n, k=32, iters=4):
    _, nb = cKDTree(X).query(X, k=k)
    for _ in range(iters):
        n = n[nb].mean(1); n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    return n


def lesion_free_colour(X, col, lesion, k=16):
    out = col.copy(); ok = lesion < 0.3
    if ok.sum() > k and (~ok).any():
        _, nb = cKDTree(X[ok]).query(X[~ok], k=k); out[~ok] = col[ok][nb].mean(1)
    return out


def mold_colour(X, rng, scale_L):
    """'Olive-green to brown': per-Gaussian blend between an olive-green and a brown Lab anchor with a smooth random
    mottle (velvety, not uniform). The anchors are our reading of the colour words (assumption)."""
    olive, brown = np.array([42.0, -6.0, 30.0]), np.array([34.0, 9.0, 22.0])
    t = rng.random(len(X)); _, nb = cKDTree(X).query(X, k=24)
    for _ in range(3):
        t = t[nb].mean(1)
    t = np.clip((t - t.min()) / (np.ptp(t) + 1e-9), 0, 1)
    lab = olive[None] * (1 - t[:, None]) + brown[None] * t[:, None]; lab[:, 0] += rng.normal(0, 2.5, len(X))
    return lab_to_srgb(lab)


def build(F, out_ply, lesion=None, lower_symptom=None, lower_symptom_rgb=None, delta_rel=0.002, upper_rgb=None):
    """F: anatomy.leaf_frame. lesion: (N,) weights of upper-surface lesions removed from the lower base colour.
    lower_symptom: (N,) weights in [0,1] of the symptom painted on the LOWER layer; lower_symptom_rgb: (N,3)."""
    v = F["v"]; X = F["X"]; L = F["L"]; N = len(X); delta = delta_rel * L
    n = smooth_normals(X, F["n"])
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64); q /= np.linalg.norm(q, axis=1, keepdims=True)
    R0 = quat_to_mat(q); S = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64))
    order = np.argsort(-S, 1); a1 = np.take_along_axis(R0, order[:, None, :1].repeat(3, 1), 2)[..., 0]
    e1 = a1 - (a1 * n).sum(1, keepdims=True) * n; bad = np.linalg.norm(e1, axis=1) < 1e-6
    e1[bad] = np.cross(n[bad], [1.0, 0, 0]); e1 /= np.linalg.norm(e1, axis=1, keepdims=True); e2 = np.cross(n, e1)
    R = np.stack([e1, e2, n], 2); qn = mat_to_quat(R)
    s_sorted = np.take_along_axis(S, order, 1); thin = np.full(N, delta / 3)
    col_up = rgb(v) if upper_rgb is None else upper_rgb
    col_lo = lesion_free_colour(X, rgb(v), lesion) if lesion is not None else rgb(v).copy()
    if lower_symptom is not None:
        col_lo = (1 - lower_symptom[:, None]) * col_lo + lower_symptom[:, None] * lower_symptom_rgb
    w = np.concatenate([v, v]).copy()
    P = np.concatenate([X + delta * n, X - delta * n])
    w["x"], w["y"], w["z"] = P.T.astype(np.float32)
    for k in range(4): w[f"rot_{k}"] = np.concatenate([qn[:, k], qn[:, k]]).astype(np.float32)
    sc = np.log(np.stack([s_sorted[:, 0], s_sorted[:, 1], np.minimum(thin, s_sorted[:, 2])], 1))
    for k in range(3): w[f"scale_{k}"] = np.concatenate([sc[:, k], sc[:, k]]).astype(np.float32)
    dc = rgb_to_dc(np.concatenate([col_up, col_lo]))
    for k in range(3): w[f"f_dc_{k}"] = dc[:, k].astype(np.float32)
    for k, nn in zip(("nx", "ny", "nz"), (n.T)):
        if k in w.dtype.names: w[k] = np.concatenate([nn, -nn]).astype(np.float32)
    PlyData([PlyElement.describe(w, "vertex")], text=False).write(out_ply)
    layers = np.concatenate([np.zeros(N, np.uint8), np.ones(N, np.uint8)]); np.save(out_ply + ".layers.npy", layers)
    rep = dict(n_upper=N, n_lower=N, delta_rel=delta_rel, thickness_scale_rel=float(delta / 3 / L),
               lower_base="original colour with upper lesions removed (unverified)",
               lower_symptom_frac=float((lower_symptom >= 0.5).mean()) if lower_symptom is not None else 0.0)
    json.dump(rep, open(out_ply + ".json", "w"), indent=1); return rep, n
