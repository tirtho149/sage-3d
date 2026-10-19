"""Geometry-aware editing of a 3D Gaussian asset (method G2: explicit, regularised position optimisation).

Representation: G = {mu_i, R_i, S_i, alpha_i, c_i}. Edited positions mu_i' = mu_i + m_i * d_i, where m_i is the
editable-region weight and d_i is found by minimising

  L = l_edit * L_edit + l_arap * L_arap + l_smooth * L_smooth + l_anchor * L_anchor

  L_edit   = mean_i m_i (d_i . n_i - h_i)^2 / L^2           target normal offset h (from KB operators, below)
             (generic arm G: mean_i m_i |d_i - g_i|^2 / L^2, g = random smooth vector field)
  L_arap   = mean_(i,j) in kNN ((|e_ij'| - |e_ij|) / |e_ij|)^2 edge-length (as-rigid-as-possible) preservation
  L_smooth = mean_(i,j) |d_i - d_j|^2 / L^2                    graph-Laplacian smoothness of the field
  L_anchor = mean_i (1 - m_i) |d_i|^2 / L^2                    keep non-target parts in place
  L = bbox diagonal (unit-free; NOT a physical length)

KB operators -> measurable target h_i (all smooth RBF fields on the surface; parameters relative to L):
  pucker   ("puckered", "blister")      zero-mean sum_k exp(-|mu-c_k|^2 / 2 s^2), c_k Poisson-disk with spacing >= 3 s
  distort  ("distorted", "deformed")    two broad RBFs of opposite sign (sigma = 0.3 L): low-frequency warp
  outgrowth("gall", "tumor-like")       few large positive bumps (galls), optionally placed at given centres
  depress  ("crater", "sunken")         negative small bumps at given centres (e.g. lesion sites) or sampled
  thicken  ("thickened")                no displacement; Gaussian scale along n_i multiplied by a factor
Rotations: each Gaussian is rotated by the minimal rotation taking its old local normal to the new one
(normals re-estimated by PCA on the same kNN set after deformation). Scales unchanged except `thicken`.
Colours/opacities never change here (appearance is a separate stage).

usage (sam3d env): deform.py <in.ply> <spec.json> <out.ply>   -> out.ply, out.ply.json (losses + metrics)
"""
import sys, json, time
import numpy as np, torch
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(p):
    return PlyData.read(p)["vertex"].data.copy()


def pca_normals(X, nb):
    P = X[nb] - X[nb].mean(1, keepdims=True); C = np.einsum("nki,nkj->nij", P, P)
    w, V = np.linalg.eigh(C); return V[:, :, 0], w


def orient(X, n, nb, mode):
    if mode == "volume":                       # outward from centroid
        s = np.sign(((X - X.mean(0)) * n).sum(1)); s[s == 0] = 1; return n * s[:, None]
    g = np.linalg.eigh(np.cov((X - X.mean(0)).T))[1][:, 0]; n = n * np.sign(n @ g + 1e-12)[:, None]
    for _ in range(10):                        # neighbour majority smoothing of the sign (curled sheets)
        s = np.sign((n[nb].mean(1) * n).sum(1)); s[s == 0] = 1; n = n * s[:, None]
    return n


def fps(X, k, rng):
    idx = [int(rng.integers(len(X)))]; d = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(k - 1):
        i = int(d.argmax()); idx.append(i); d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return np.array(idx)


def poisson_disk(X, r, rng):
    """Greedy Poisson-disk subset of points X with minimum spacing r (random order)."""
    order = rng.permutation(len(X)); tree = cKDTree(X); taken = np.zeros(len(X), bool); blocked = np.zeros(len(X), bool)
    for i in order:
        if blocked[i]: continue
        taken[i] = True; blocked[tree.query_ball_point(X[i], r)] = True
    return X[taken]


def rbf(X, C, s):
    return np.exp(-((X[:, None, :] - C[None]) ** 2).sum(-1) / (2 * s * s))


def target_field(X, n, ops, L, rng):
    """Returns h (normal offsets), g (generic vector field or None), thicken factor, op log."""
    h = np.zeros(len(X)); g = None; thick = 1.0; log = []
    for op in ops:
        t = op["op"]
        if t == "pucker":
            # distinct blisters: Poisson-disk centres (min spacing 3 s), zero-mean field (no net lift of the organ)
            # v2: optional placement weights w_i in [0, 1] (weights_npy): centres are drawn only where w_i >= 0.5 and the
            # field is multiplied by w (evidence-constrained placement, see geometry_editor/placement.py)
            s = op.get("sigma", 0.035) * L; W = np.load(op["weights_npy"]) if "weights_npy" in op else None
            C = poisson_disk(X if W is None else X[W >= 0.5], 3 * s, rng)
            f = np.zeros(len(X))
            for c0 in range(0, len(C), 64): f += rbf(X, C[c0:c0 + 64], s).sum(1)
            f = f - f.mean(); f = f / (np.percentile(np.abs(f), 99) + 1e-9)
            if W is not None: f = f * W
            h += op.get("amp", 0.02) * L * f; log.append(dict(op=t, n_bumps=int(len(C)), sigma=s, weighted=W is not None))
        elif t == "distort":
            C = X[fps(X, 2, rng)]; s = op.get("sigma", 0.3) * L
            f = rbf(X, C, s) @ np.array([1.0, -1.0]); h += op.get("amp", 0.06) * L * f; log.append(dict(op=t, sigma=s))
        elif t in ("outgrowth", "depress"):
            sgn = 1.0 if t == "outgrowth" else -1.0; s = op.get("sigma", 0.08 if t == "outgrowth" else 0.02) * L
            C = np.asarray(op["centres"]) if "centres" in op else X[fps(X, op.get("n", 3), rng)]
            h += sgn * op.get("amp", 0.08 if t == "outgrowth" else 0.01) * L * rbf(X, C, s).max(1); log.append(dict(op=t, n=len(C), sigma=s))
        elif t == "vector":                    # v2: precomputed vector target (e.g. directional curl, curl.py)
            g = np.load(op["target_npy"]) if g is None else g + np.load(op["target_npy"]); log.append(dict(op=t, target=op["target_npy"], kb_term=op.get("kb_term")))
        elif t == "thicken":
            thick *= op.get("factor", 2.0); log.append(dict(op=t, factor=thick))
        elif t == "generic":                   # arm G: random smooth vector field, RMS set by `rms` (fraction of L)
            C = X[fps(X, op.get("n", 6), rng)]; W = rng.normal(size=(len(C), 3)); s = op.get("sigma", 0.15) * L
            gg = np.zeros_like(X)
            for c0 in range(0, len(C), 64): gg += rbf(X, C[c0:c0 + 64], s) @ W[c0:c0 + 64]
            g = gg * (op.get("rms", 0.02) * L / (np.sqrt((gg ** 2).sum(1).mean()) + 1e-12)); log.append(dict(op=t, n=len(C), sigma=s))
        else:
            raise ValueError(f"unknown op {t}")
    return h, g, thick, log


def quat_mul(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], 1)


def min_rot_quat(a, b):
    """Unit quaternion (w,x,y,z) rotating unit vectors a -> b (minimal rotation)."""
    c = np.cross(a, b); d = (a * b).sum(1); q = np.concatenate([1 + d[:, None], c], 1)
    bad = d < -0.9999; q[bad] = np.array([0, 1, 0, 0])
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def quat_to_mat(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1), np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                     np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], 1)


def run(inp, spec, out):
    t0 = time.time(); v = load(inp); rng = np.random.default_rng(spec.get("seed", 0))
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64); L = float(np.linalg.norm(X.max(0) - X.min(0)))
    tree = cKDTree(X); _, nb = tree.query(X, k=16)
    ev = np.linalg.eigvalsh(np.cov((X - X.mean(0)).T)); mode = spec.get("mode") or ("sheet" if ev[0] / ev[2] < 0.05 else "volume")
    n, _ = pca_normals(X, nb); n = orient(X, n, nb, mode)
    _, nb32 = tree.query(X, k=32)                         # smooth the normal field (margin PCA normals are unreliable)
    for _ in range(spec.get("normal_smooth_iters", 3)):
        n = n[nb32].mean(1); n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    if "normals_npy" in spec:                             # v2: anatomy-oriented normals (adaxial side, anatomy.py)
        n = np.load(spec["normals_npy"]).astype(np.float64)
    m = np.ones(len(X))                                   # editable region: whole organ unless a region is given
    if "region_weights_npy" in spec:
        m = np.load(spec["region_weights_npy"]).astype(np.float64)
    elif "region_centres" in spec:
        m = rbf(X, np.asarray(spec["region_centres"]), spec.get("region_sigma", 0.15) * L).max(1)
    h, g, thick, oplog = target_field(X, n, spec["ops"], L, rng)
    # optimisation
    T = lambda a: torch.as_tensor(a, dtype=torch.float32, device=DEV)
    Xt, nt, ht, mt = T(X), T(n), T(h), T(m); E = T(nb[:, 1:9]).long(); I = torch.arange(len(X), device=DEV)[:, None].expand_as(E)
    e0 = (Xt[I] - Xt[E]).norm(dim=-1).clamp_min(1e-9)
    if spec.get("arap_coarse"):                            # v2: coarse-scale ARAP on a 1/16 subsample (8-NN there spans
        sub = np.sort(rng.choice(len(X), max(len(X) // 16, 100), replace=False))   # ~4x the fine spacing), so that
        _, nbc = cKDTree(X[sub]).query(X[sub], k=9)                                # bending stays near-isometric at
        Ec = torch.as_tensor(sub[nbc[:, 1:]], device=DEV).long(); Ic = torch.as_tensor(sub, device=DEV).long()[:, None].expand_as(Ec)  # the scale of tissue, not noise
        ec0 = (Xt[Ic] - Xt[Ec]).norm(dim=-1).clamp_min(1e-9)
    d = torch.nn.Parameter(T(g) if g is not None else nt * ht[:, None])
    lam = {**dict(edit=1.0, arap=0.1, smooth=0.01, anchor=1.0), **spec.get("lambdas", {})}
    opt = torch.optim.Adam([d], lr=1e-3 * L); hist = []
    for it in range(spec.get("iters", 300)):
        dm = mt[:, None] * d; Y = Xt + dm
        L_edit = ((mt * ((d * nt).sum(1) - ht) ** 2).mean() if g is None else (mt[:, None] * (d - T(g)) ** 2).sum(1).mean()) / L ** 2
        L_arap = (((Y[I] - Y[E]).norm(dim=-1) - e0) / e0).pow(2).mean()
        if spec.get("arap_coarse"): L_arap = L_arap + spec["arap_coarse"] * (((Y[Ic] - Y[Ec]).norm(dim=-1) - ec0) / ec0).pow(2).mean()
        L_smooth = (dm[I] - dm[E]).pow(2).sum(-1).mean() / L ** 2
        L_anchor = ((1 - mt) * d.pow(2).sum(1)).mean() / L ** 2
        loss = lam["edit"] * L_edit + lam["arap"] * L_arap + lam["smooth"] * L_smooth + lam["anchor"] * L_anchor
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 50 == 0 or it == spec.get("iters", 300) - 1:
            hist.append(dict(it=it, total=float(loss), edit=float(L_edit), arap=float(L_arap), smooth=float(L_smooth), anchor=float(L_anchor)))
    D = (mt[:, None] * d).detach().cpu().numpy().astype(np.float64); Y = X + D
    # rotations: minimal rotation old normal -> new normal (same neighbourhoods)
    n2, _ = pca_normals(Y, nb); n2 = n2 * np.sign((n2 * n).sum(1) + 1e-12)[:, None]
    for _ in range(spec.get("normal_smooth_iters", 3)):
        n2 = n2[nb32].mean(1); n2 /= np.linalg.norm(n2, axis=1, keepdims=True).clip(1e-12)
    q = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1).astype(np.float64); q /= np.linalg.norm(q, axis=1, keepdims=True)
    qn = quat_mul(min_rot_quat(n, n2), q)
    w = v.copy(); w["x"], w["y"], w["z"] = Y.T.astype(np.float32)
    for k in range(4): w[f"rot_{k}"] = qn[:, k].astype(np.float32)
    if thick != 1.0:                                      # scale the Gaussian axis most aligned with the normal
        Rg = quat_to_mat(qn); al = np.abs(np.einsum("nij,ni->nj", Rg, n2)); ax = al.argmax(1); ok = al.max(1) > 0.8
        for k in range(3):                                # only where the Gaussian's normal axis is unambiguous
            sel = (ax == k) & ok; w[f"scale_{k}"][sel] = (w[f"scale_{k}"][sel] + np.log(thick)).astype(np.float32)
    PlyData([PlyElement.describe(w, "vertex")], text=False).write(out)
    # realised-edit metrics
    hn = (D * n).sum(1); ed = np.linalg.norm(Y[nb[:, 1:9]] - Y[:, None], axis=2) / np.linalg.norm(X[nb[:, 1:9]] - X[:, None], axis=2).clip(1e-9)
    ang = np.degrees(np.arccos(np.clip(np.abs((n2 * n).sum(1)), 0, 1)))
    rep = dict(input=inp, spec=spec, mode=mode, L_bbox_diag=L, oplog=oplog, loss_history=hist, lambdas=lam, sec=time.time() - t0,
               target_normal_rms_rel=float(np.sqrt((h ** 2).mean()) / L) if g is None else None,
               realised_normal_rms_rel=float(np.sqrt((hn ** 2).mean()) / L),
               edit_fidelity_corr=float(np.corrcoef(hn, h)[0, 1]) if g is None and h.std() > 0 else None,
               disp_rel=dict(median=float(np.median(np.linalg.norm(D, axis=1)) / L), p95=float(np.percentile(np.linalg.norm(D, axis=1), 95) / L)),
               edge_stretch=dict(median_abs=float(np.median(np.abs(ed - 1))), p95_abs=float(np.percentile(np.abs(ed - 1), 95))),
               normal_change_deg=dict(median=float(np.median(ang)), p95=float(np.percentile(ang, 95))),
               displacement_in_region_frac=float((np.linalg.norm(D, axis=1) * m).sum() / (np.linalg.norm(D, axis=1).sum() + 1e-12)),
               thicken_factor=thick, thicken_applied_frac=float(ok.mean()) if thick != 1.0 else 0.0, n=len(X))
    json.dump(rep, open(out + ".json", "w"), indent=1); return rep


if __name__ == "__main__":
    r = run(sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3])
    print(json.dumps({k: r[k] for k in ("mode", "target_normal_rms_rel", "realised_normal_rms_rel", "edit_fidelity_corr", "disp_rel", "edge_stretch", "normal_change_deg", "sec")}))
