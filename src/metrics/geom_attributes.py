"""KB geometric-attribute scores of an edited asset relative to the unedited asset (brief s11/s13: symptom recall).

Given matched Gaussians (gaussian_diff.match) and the ORIGINAL surface normals n_i, the out-of-plane displacement
h_i = d_i . n_i is split by spatial scale with normalised Gaussian smoothing on a voxel grid:
  low  = S_{0.12L}[h]                    -> "distorted"  score = RMS(low)/L
  mid  = S_{0.02L}[h] - S_{0.12L}[h]     -> "puckered"   score = RMS(mid)/L   (blister scale)
  high = h - S_{0.02L}[h]                -> jitter       score = RMS(high)/L  (incoherent, artefact)
  thickening = median ratio of Gaussian extent along each state's own surface normal (edited / original) -> "thickened"
Band edges and operator scales were fixed on a synthetic flat sheet (tests/test_geometry_ops.py), not on specimens.
  tangential = RMS of the in-plane displacement / L
Scales are fixed fractions of the bbox diagonal L and were set from the operator definitions (not tuned on results).
usage (sam3d env): geom_attributes.py <orig.ply> <edited.ply> [out.json]
"""
import sys, json
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from plyfile import PlyData
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gaussian_diff import load, match


def smooth(X, f, s, vox):
    lo = X.min(0) - 3 * s; idx = np.floor((X - lo) / vox).astype(int); shp = idx.max(0) + 2 + int(3 * s / vox)  # pad both sides
    num = np.zeros(shp); den = np.zeros(shp); np.add.at(num, tuple(idx.T), f); np.add.at(den, tuple(idx.T), 1.0)
    num = ndimage.gaussian_filter(num, s / vox); den = ndimage.gaussian_filter(den, s / vox)
    c = ((X - lo) / vox - 0.5).T
    return ndimage.map_coordinates(num, c, order=1, mode='nearest') / np.maximum(ndimage.map_coordinates(den, c, order=1, mode='nearest'), 1e-9)


def quat_to_mat(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1), np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                     np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], 1)


def normals(X):
    _, nb = cKDTree(X).query(X, k=16); P = X[nb] - X[nb].mean(1, keepdims=True)
    n = np.linalg.eigh(np.einsum("nki,nkj->nij", P, P))[1][:, :, 0]
    _, nb32 = cKDTree(X).query(X, k=32)
    g = np.linalg.eigh(np.cov((X - X.mean(0)).T))[1][:, 0]; n = n * np.sign(n @ g + 1e-12)[:, None]
    for _ in range(10):
        s = np.sign((n[nb].mean(1) * n).sum(1)); s[s == 0] = 1; n = n * s[:, None]
    for _ in range(3):
        n = n[nb32].mean(1); n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    return n


def attributes(orig, edited):
    A, B = load(orig), load(edited); ia, jb, _ = match(A, B)
    X = A["xyz"]; L = float(np.linalg.norm(X.max(0) - X.min(0))); n = normals(X)[ia]
    d = B["xyz"][jb] - X[ia]; h = (d * n).sum(1); tang = d - h[:, None] * n; Xm = X[ia]
    vox = 0.005 * L; low = smooth(Xm, h, 0.12 * L, vox * 4); mid_s = smooth(Xm, h, 0.02 * L, vox)
    mid = mid_s - low; high = h - mid_s
    ext = lambda logs, q, nn: np.sqrt((np.einsum("nij,ni->nj", quat_to_mat(q), nn) ** 2 * np.exp(2 * logs)).sum(1))
    nB = normals(B["xyz"])[jb]; nB = nB * np.sign((nB * n).sum(1) + 1e-12)[:, None]     # extent along each state's own normal
    thick = np.median(ext(B["logs"][jb], B["q"][jb], nB) / ext(A["logs"][ia], A["q"][ia], n))
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)) / L)
    # blister count: local maxima of the mid-band field (radius 0.05 L) above max(0.3 * its 99th percentile, 0.003 L).
    # Puckering = many distinct blisters; a generic smooth warp of equal energy has few.
    rng = np.random.default_rng(0); sub = rng.choice(len(Xm), min(30000, len(Xm)), replace=False); Xs, ms = Xm[sub], mid[sub]
    nb = cKDTree(Xs).query_ball_point(Xs, 0.05 * L); thr = max(0.3 * np.percentile(np.abs(ms), 99), 0.003 * L)   # absolute floor: 0.3% of L
    peaks = sum(1 for i, nn in enumerate(nb) if ms[i] > thr and ms[i] >= ms[nn].max())
    # band-pass detectors for localised relief (fixed on the synthetic sheet):
    #   outgrowth  = 99.5th pct of  S_{0.03L}[h] - S_{0.3L}[h]   (gall-sized positive protrusions, broad warps removed)
    #   depression = -0.5th pct of  S_{0.005L}[h] - S_{0.04L}[h] (pit-sized negative dips)
    b_out = smooth(Xm, h, 0.03 * L, vox * 2) - smooth(Xm, h, 0.3 * L, vox * 6)
    b_dep = smooth(Xm, h, 0.005 * L, vox) - smooth(Xm, h, 0.04 * L, vox * 2)
    outgrowth = float(max(0.0, np.percentile(b_out, 99.5)) / L); depression = float(max(0.0, -np.percentile(b_dep, 0.5)) / L)
    sp = np.median(cKDTree(Xm).query(Xm[sub[:5000]], k=2)[0][:, 1]); area = len(Xm) * sp * sp      # sheet area estimate
    density = peaks / (area * len(sub) / len(Xm) / (0.1 * L) ** 2) if area > 0 else 0.0              # blisters per (0.1 L)^2, on the subsample
    return dict(distorted=rms(low), puckered=rms(mid), blister_count=int(peaks), blister_density=float(density), outgrowth=outgrowth, depression=depression, jitter=rms(high), thickened=float(thick), tangential=rms(np.linalg.norm(tang, axis=1)),
                n_matched=int(len(ia)), L=L)


if __name__ == "__main__":
    r = attributes(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3: json.dump(r, open(sys.argv[3], "w"), indent=1)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))
