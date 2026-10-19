"""Structural measures of an edited asset against its unedited asset (matched Gaussian indices: the v2 editors never
prune or reorder Gaussians; added Gaussians, e.g. galls, are appended after the original N).

Leaf roll / curl (needs the leaf frame of the ORIGINAL asset, anatomy.leaf_frame):
  roll angle per u-bin and side: theta = atan2((c_margin - c_mid) . n_mid, (c_margin - c_mid) . t_side)
     c_mid = centroid of |r| < 0.15, c_margin = centroid of r > 0.8 (or < -0.8), n_mid = adaxial midrib normal,
     t_side = lateral tangent at the midrib. theta > 0: margin lies toward the ADAXIAL side (upward / involute);
     theta < 0: toward the abaxial side (downward / revolute). Median over bins (end bins excluded).
  transverse curvature kappa_t: quadratic fit h(s) = a + b s + c s^2 of the cross-section (s along t_+, h along n_mid);
     kappa_t = 2c, reported dimensionless as kappa_t * half-width. Same sign convention.
  mean-curvature field H_i: quadratic patch fit z = a x^2 + b xy + c y^2 in each point's tangent frame (k = 24);
     H = a + c (sign: + = concave toward adaxial). Reported as H * L.
Local deformation preservation:
  local non-rigidity eps_i = sqrt(sum_j |(y_j - y_i) - R_i (x_j - x_i)|^2 / sum_j |x_j - x_i|^2) with R_i the best
     rotation of the k = 16 neighbourhood (Kabsch). 0 = locally rigid (shape of small patches preserved).
  edge strain |e'/e - 1| over 8-NN edges (median, p95): leaf tissue does not stretch much under rolling.
  fine-relief preservation: corr(h_fine before, h_fine after) with h_fine_i = (x_i - mean_64NN(x)) . n_i, i.e. small
     bumps already present in the reconstruction (from the photo) survive the edit.
Input view (independent REAL reference): silhouette IoU of the asset rendered at the corrected input camera vs the
  real photo's organ mask, before and after the edit. An edit should not contradict what the photo shows.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def oriented_normals(Y, ref, k=16, smooth=2):
    tree = cKDTree(Y); _, nb = tree.query(Y, k=k); P = Y[nb] - Y[nb].mean(1, keepdims=True)
    n = np.linalg.eigh(np.einsum("nki,nkj->nij", P, P))[1][:, :, 0]; n *= np.sign((n * ref).sum(1) + 1e-12)[:, None]
    for _ in range(smooth):
        n = n[nb].mean(1); n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    return n


def roll_metrics(F, Y, bins=12):
    X, r, u, n0, t0 = F["X"], F["r"], F["u"], F["n"], F["t_lat"]
    edges = np.quantile(u, np.linspace(0, 1, bins + 1)); b = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, bins - 1)
    th = {+1: [], -1: []}; kap = []; prof = []           # prof: (bin, mean roll of the sides measured in that bin)
    for i in range(1, bins - 1):
        sel = b == i; mid = sel & (np.abs(r) < 0.15)
        if mid.sum() < 20: continue
        cm = Y[mid].mean(0); nm = n0[mid].mean(0)
        # midrib normal in the edited state: plane fit of the midrib neighbourhood, oriented like the original
        Q = Y[sel & (np.abs(r) < 0.35)] - cm
        if len(Q) < 30: continue
        nm2 = np.linalg.eigh(Q.T @ Q)[1][:, 0]; nm2 *= np.sign(nm2 @ nm + 1e-12)
        for sd in (+1, -1):
            inner = sel & (sd * r > 0.1) & (sd * r < 0.35); mar = sel & (sd * r > 0.8)
            if inner.sum() < 10 or mar.sum() < 10: continue
            ts = (Y[inner].mean(0) - cm); ts -= (ts @ nm2) * nm2; ts /= np.linalg.norm(ts) + 1e-12
            dv = Y[mar].mean(0) - cm; th[sd].append(np.degrees(np.arctan2(dv @ nm2, dv @ ts))); prof.append((i, sd, th[sd][-1]))
        tp = Y[sel & (r > 0.1) & (r < 0.35)].mean(0) - Y[sel & (r < -0.1) & (r > -0.35)].mean(0) if ((sel & (r > 0.1) & (r < 0.35)).sum() > 10 and (sel & (r < -0.1) & (r > -0.35)).sum() > 10) else None
        if tp is not None:
            tp -= (tp @ nm2) * nm2; tp /= np.linalg.norm(tp) + 1e-12
            P = Y[sel] - cm; s = P @ tp; h = P @ nm2; half = np.percentile(np.abs(s), 95)
            if half > 0:
                c2 = np.polyfit(s / half, h / half, 2)[0]; kap.append(2 * c2)
    allth = th[+1] + th[-1]
    return dict(roll_deg_median=float(np.median(allth)) if allth else None, roll_deg_side_plus=float(np.median(th[+1])) if th[+1] else None,
                roll_deg_side_minus=float(np.median(th[-1])) if th[-1] else None, kappa_t_halfwidth_median=float(np.median(kap)) if kap else None,
                n_bins=len(kap),
                roll_profile=[dict(bin=int(i), mean_deg=float(np.mean([t for j, _, t in prof if j == i])))
                              for i in sorted({j for j, _, _ in prof})])


def mean_curvature(Y, n, k=24, sub=None):
    idx = np.arange(len(Y)) if sub is None else sub; tree = cKDTree(Y); _, nb = tree.query(Y[idx], k=k)
    nn = n[idx]; a = np.cross(nn, np.where(np.abs(nn[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])); a /= np.linalg.norm(a, axis=1, keepdims=True)
    bvec = np.cross(nn, a); P = Y[nb] - Y[idx][:, None]
    x = np.einsum("nkj,nj->nk", P, a); y = np.einsum("nkj,nj->nk", P, bvec); z = np.einsum("nkj,nj->nk", P, nn)
    A = np.stack([x * x, x * y, y * y, x, y], -1); sol = np.linalg.lstsq if False else None
    AtA = np.einsum("nki,nkj->nij", A, A) + 1e-12 * np.eye(5); Atz = np.einsum("nki,nk->ni", A, z)
    c = np.linalg.solve(AtA, Atz[..., None])[..., 0]
    return c[:, 0] + c[:, 2]


def _patches(X, L, r_rel, n_centres, m, seed):
    """FPS patch centres; each patch = up to m random points within radius r_rel * L (fixed physical scale, so the
    measure is not dominated by the sub-millimetre noise of a reconstructed sheet)."""
    from geometry_editor.anatomy import fps
    rng = np.random.default_rng(seed); c = fps(X[rng.choice(len(X), min(len(X), 40000), replace=False)], n_centres, seed)
    sub = rng.choice(len(X), min(len(X), 40000), replace=False); C = X[sub][c]
    tree = cKDTree(X); P = []
    for cc in C:
        nb = tree.query_ball_point(cc, r_rel * L)
        if len(nb) >= 12: P.append(np.asarray(nb)[rng.choice(len(nb), min(m, len(nb)), replace=False)])
    return P


def nonrigidity(X, Y, L, r_rel=0.05, n_centres=1500, m=64, seed=0):
    """Patch non-rigidity at a fixed scale (radius 0.05 L): residual after the best rigid motion, relative to the
    patch extent. 0 = patch moved rigidly (its shape, including small bumps, preserved)."""
    out = []
    for nb in _patches(X, L, r_rel, n_centres, m, seed):
        P = X[nb] - X[nb].mean(0); Q = Y[nb] - Y[nb].mean(0); U, S, Vt = np.linalg.svd(Q.T @ P)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt
        out.append(np.sqrt(((Q - P @ R.T) ** 2).sum() / (P ** 2).sum()))
    return np.array(out)


def edge_strain(X, Y, k=9):
    _, nb = cKDTree(X).query(X, k=k); e0 = np.linalg.norm(X[nb[:, 1:]] - X[:, None], axis=2).clip(1e-12)
    e1 = np.linalg.norm(Y[nb[:, 1:]] - Y[:, None], axis=2); s = np.abs(e1 / e0 - 1); return float(np.median(s)), float(np.percentile(s, 95))


def _kabsch(P, Q):
    U, S, Vt = np.linalg.svd(Q.T @ P); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); return U @ D @ Vt


def fine_relief_corr(X, Y, L, r_rel=0.04, n_centres=1500, m=128, seed=0):
    """Relief at the < 0.04 L scale (bumps, blisters, puckering captured from the photo): residual of each patch
    point from the patch's best-fit QUADRIC h = f(s, t) (so smooth bending, e.g. a curl, is not counted as lost
    relief). Normals of the edited patch are aligned through the patch's best rotation. Correlation before/after."""
    h0, h1 = [], []
    for nb in _patches(X, L, r_rel, n_centres, m, seed):
        P = X[nb] - X[nb].mean(0); Q = Y[nb] - Y[nb].mean(0); R = _kabsch(P, Q)
        Ep = np.linalg.eigh(P.T @ P)[1]; npn = Ep[:, 0] * np.sign(Ep[np.argmax(np.abs(Ep[:, 0])), 0] + 1e-12)
        for Z, h, nref in ((P, h0, npn), (Q, h1, R @ npn)):
            E = np.linalg.eigh(Z.T @ Z)[1]; n, a, b = E[:, 0], E[:, 2], E[:, 1]; n = n * np.sign(n @ nref + 1e-12)
            s_, t_, z = Z @ a, Z @ b, Z @ n; A = np.stack([s_ * s_, s_ * t_, t_ * t_, s_, t_, np.ones_like(s_)], 1)
            h.append(z - A @ np.linalg.lstsq(A, z, rcond=None)[0])
    return float(np.corrcoef(np.concatenate(h0), np.concatenate(h1))[0, 1])


def pair_strain(X, Y, L, lo=0.015, hi=0.025, n=40000, seed=0):
    """Tissue stretch at a fixed scale: relative length change of random point pairs 0.015-0.025 L apart."""
    rng = np.random.default_rng(seed); i = rng.choice(len(X), n); tree = cKDTree(X); out = []
    for a in i[:8000]:
        nb = tree.query_ball_point(X[a], hi * L)
        if len(nb) < 2: continue
        b = nb[rng.integers(len(nb))]; d0 = np.linalg.norm(X[a] - X[b])
        if d0 < lo * L: continue
        out.append(abs(np.linalg.norm(Y[a] - Y[b]) / d0 - 1))
    out = np.array(out); return float(np.median(out)), float(np.percentile(out, 95))


def deformation_report(F, Y, n_sub=20000, seed=0):
    X = F["X"]; L = F["L"]; rng = np.random.default_rng(seed); sub = rng.choice(len(X), min(n_sub, len(X)), replace=False)
    eps = nonrigidity(X, Y, L, r_rel=0.02); sm, sp = pair_strain(X, Y, L)
    n1 = oriented_normals(Y, F["n"]); H0 = mean_curvature(X, F["n"], sub=sub) * L; H1 = mean_curvature(Y, n1, sub=sub) * L
    return dict(nonrigidity_02L_median=float(np.median(eps)), nonrigidity_02L_p95=float(np.percentile(eps, 95)), strain_02L_median=sm, strain_02L_p95=sp,
                fine_relief_corr=fine_relief_corr(X, Y, L), meanH_L_median_before=float(np.median(H0)), meanH_L_median_after=float(np.median(H1)),
                meanH_L_shift_median=float(np.median(H1 - H0)), frac_points_H_increased=float((H1 > H0).mean()),
                disp_rms_rel=float(np.sqrt(((Y - X) ** 2).sum(1).mean()) / L))


def input_view_iou(ply, sid):
    """Silhouette IoU at the corrected input camera vs the REAL photo's organ mask (independent of the edit)."""
    sys.path.insert(0, f"{ROOT}/src"); os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    from camera_alignment.align import render_at_input
    rgb, a = render_at_input(ply, sid); m = np.asarray(Image.open(f"{ROOT}/data/inputs/{sid}/0.png").convert("L")) > 127
    p = a > 0.5; return float((p & m).sum() / max((p | m).sum(), 1)), rgb
