"""Organ anatomy for a single-photo SAM 3D asset: input camera, surface sides, a flattened leaf frame, an ear frame.

Everything here is inferred from the asset and the photo, and every inference that cannot be checked is logged in
`assumptions` (returned with each frame) so that downstream reports can state it.

Leaf (sheet) frame
  * camera: the A3*-corrected input camera, expressed in the asset's object coordinates (Umeyama fit between the
    object-space means and their posed, camera-space positions; exact up to float error).
  * adaxial side: the surface side that faces the input camera is labelled ADAXIAL (upper). This is an ASSUMPTION
    (most SAGE leaf photos show the upper surface); for Tomato Leaf Mold it is supported by the photo itself, since
    the KB places the pale spots, which the photo shows, on the upper side. Normals are seeded by the camera at
    Gaussians visible in the input view and propagated over the sheet by neighbour-majority sign smoothing, so a
    margin that curls away from the camera keeps the orientation of the lamina it belongs to.
  * flattened coordinates (u, v): landmark Isomap (geodesic distances on a kNN graph, classical MDS to 2D).
    u = major axis of the flattened lamina (approximate midrib direction), v = lateral coordinate.
    Midrib line = per-u-bin midpoint of the v extent; half-width w(u); r = (v - mid(u)) / w(u) in [-1, 1].
    Tip vs base along u is NOT inferred (the petiole is not reliably reconstructed): logged as unknown.
Ear (volume) frame
  * axis = first principal axis; radial outward normal; kernel tissue = Gaussians whose colour is closer to the
    kernel colour cluster than to husk/background clusters (k-means on Lab, cluster chosen by highest b*/L, i.e.
    yellow-white). Gall tissue already present in the photo (grey-white, low chroma) is labelled separately.
"""
import os, sys, json
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from plyfile import PlyData

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
C0 = 0.28209479177387814


def load_vertices(ply):
    return PlyData.read(ply)["vertex"].data.copy()


def xyz(v):
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)


def rgb(v):
    return np.clip(0.5 + C0 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1), 0, 1)


def rgb_to_dc(c):
    return (np.clip(c, 0, 1) - 0.5) / C0


def srgb_to_lab(c):
    c = np.clip(c, 0, 1); l = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    X = l @ M.T / np.array([0.95047, 1.0, 1.08883]); f = np.where(X > 0.008856, np.cbrt(X), 7.787 * X + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def umeyama(A, B):
    """Similarity B ~ s R A + t."""
    ma, mb = A.mean(0), B.mean(0); a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a / len(A)); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / a.var(0).sum(); return s, R, mb - s * R @ ma


def input_camera(sid, ply=None):
    """Camera centre (object coords), similarity (s, R, t) object->camera, and the input-view visibility of each
    Gaussian of `ply` (default: original splat). Uses the A3*-corrected pose exactly as render_at_input."""
    import torch
    sys.path.insert(0, f"{ROOT}/src"); os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    from camera_alignment.align import posed_gaussians, transform, DEV
    from gsplat import rasterization
    run_dir = f"{ROOT}/results/pilot/sam3d/{sid}/seed0"; ply = ply or f"{run_dir}/splat_ge.ply"
    A = json.load(open(f"{ROOT}/results/alignment/{sid}/alignment.json"))
    G, _ = posed_gaussians(run_dir, ply); m = A["metrics"]
    sel = "A3" if m["A3_render_compare"]["iou"] > m["A2_closed_form"]["iou"] else "A2"; pc = A["pose_correction"][sel]
    c = torch.tensor(A["pose_correction"]["centroid_obj_cam"], device=DEV)
    G = transform(G, torch.tensor(pc["rot_axis_angle"], device=DEV), torch.tensor(pc["t"], device=DEV), torch.tensor(pc["log_scale"], device=DEV), c)
    Xc = G["means"].detach().cpu().numpy().astype(np.float64); Xo = xyz(load_vertices(ply))
    s, R, t = umeyama(Xo, Xc); res = np.linalg.norm(s * Xo @ R.T + t - Xc, axis=1).max() / np.linalg.norm(Xc.max(0) - Xc.min(0))
    cam_o = R.T @ (-t) / s
    # visibility at the input view: projected Gaussian centre in front of the rendered depth (+ tolerance)
    W, H = A["W"], A["H"]; K = np.array(A["K"], np.float64)
    with torch.no_grad():
        out, al, _ = rasterization(G["means"], G["quats"], G["scales"], G["op"], G["rgb"], torch.eye(4, device=DEV)[None],
                                   torch.tensor(K, dtype=torch.float32, device=DEV)[None], W, H, render_mode="RGB+ED")
    depth = out[0, ..., 3].cpu().numpy(); alpha = al[0, ..., 0].cpu().numpy()
    uvw = Xc @ K.T; px = np.round(uvw[:, 0] / uvw[:, 2]).astype(int); py = np.round(uvw[:, 1] / uvw[:, 2]).astype(int)
    ok = (px >= 0) & (px < W) & (py >= 0) & (py < H) & (Xc[:, 2] > 0)
    vis = np.zeros(len(Xc), bool); L = np.linalg.norm(Xc.max(0) - Xc.min(0))
    vis[ok] = (Xc[ok, 2] <= depth[py[ok], px[ok]] + 0.01 * L) & (alpha[py[ok], px[ok]] > 0.5)
    return dict(cam_obj=cam_o, s=s, R=R, t=t, fit_residual_rel=float(res), visible=vis, pose=sel, W=W, H=H, K=K, Xc=Xc)


def pca_normals(X, nb):
    P = X[nb] - X[nb].mean(1, keepdims=True); w, V = np.linalg.eigh(np.einsum("nki,nkj->nij", P, P)); return V[:, :, 0]


def adaxial_normals(X, cam, visible, k=16, iters=30):
    """Normals oriented to the camera-facing (assumed adaxial) side, propagated by neighbour majority."""
    tree = cKDTree(X); _, nb = tree.query(X, k=k); n = pca_normals(X, nb)
    seed = np.sign(((cam - X) * n).sum(1)); seed[seed == 0] = 1; n = n * seed[:, None]
    # independent check: propagate from the camera seed WITHOUT pinning the visible Gaussians and measure how many
    # visible Gaussians still face the camera (a consistently orientable sheet keeps ~1; a fold or a ring does not)
    m = n.copy()
    for _ in range(iters):
        s = np.sign((m[nb].mean(1) * m).sum(1)); s[s == 0] = 1; m = m * s[:, None]
    free_agree = float(((m[visible] * (cam - X[visible])).sum(1) > 0).mean()) if visible.any() else float("nan")
    fixed = visible.copy()
    for _ in range(iters):          # unseen Gaussians follow their neighbours; seen ones keep the camera sign
        s = np.sign((n[nb].mean(1) * n).sum(1)); s[s == 0] = 1; s[fixed] = 1; n = n * s[:, None]
    _, nb32 = tree.query(X, k=32)
    for _ in range(3):
        n = n[nb32].mean(1); n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    return n, nb, free_agree


def fps(X, k, seed=0):
    rng = np.random.default_rng(seed); idx = [int(rng.integers(len(X)))]; d = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(k - 1):
        i = int(d.argmax()); idx.append(i); d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return np.array(idx)


def isomap2d(X, n_land=2500, k=10):
    """Landmark Isomap to 2D. Returns (UV for all points, stress = 1 - corr(geodesic, embedded distance))."""
    li = fps(X, min(n_land, len(X))); Y = X[li]; tree = cKDTree(Y); d, nb = tree.query(Y, k=k + 1)
    rows = np.repeat(np.arange(len(Y)), k); G = coo_matrix((d[:, 1:].ravel(), (rows, nb[:, 1:].ravel())), shape=(len(Y),) * 2)
    G = G.maximum(G.T); ncomp, lab = connected_components(G, directed=False)
    main = np.bincount(lab).argmax(); keep = lab == main
    D = dijkstra(G.tocsr(), directed=False)[np.ix_(keep, keep)]
    n = D.shape[0]; J = np.eye(n) - 1.0 / n; B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B); Z = V[:, -2:] * np.sqrt(np.maximum(w[-2:], 0)); Z = Z[:, ::-1]
    E = np.linalg.norm(Z[:, None] - Z[None], axis=2); iu = np.triu_indices(n, 1)
    stress = float(1 - np.corrcoef(D[iu], E[iu])[0, 1])
    # all points: inverse-distance average of the embedding of their 4 nearest landmarks + tangent-plane offset
    Yk = Y[keep]; t2 = cKDTree(Yk); dd, ii = t2.query(X, k=4); w4 = 1 / np.maximum(dd, 1e-9); w4 /= w4.sum(1, keepdims=True)
    UV = (Z[ii] * w4[..., None]).sum(1)
    return UV, stress, dict(n_landmarks=int(n), n_components=int(ncomp))


def leaf_frame(sid, ply=None, bins=24):
    ply = ply or f"{ROOT}/results/pilot/sam3d/{sid}/seed0/splat_ge.ply"; v = load_vertices(ply); X = xyz(v)
    L = float(np.linalg.norm(X.max(0) - X.min(0))); cam = input_camera(sid, ply)
    n, nb, agree = adaxial_normals(X, cam["cam_obj"], cam["visible"])
    UV, stress, iso = isomap2d(X)
    UV = UV - UV.mean(0); ev, evec = np.linalg.eigh(np.cov(UV.T)); UV = UV @ evec[:, ::-1]    # u = major axis
    u, vv = UV[:, 0], UV[:, 1]
    edges = np.quantile(u, np.linspace(0, 1, bins + 1)); b = np.clip(np.searchsorted(edges, u, side="right") - 1, 0, bins - 1)
    mid = np.zeros(bins); half = np.zeros(bins)
    for i in range(bins):
        s = vv[b == i]
        if len(s) > 10: lo, hi = np.percentile(s, [1, 99]); mid[i] = (lo + hi) / 2; half[i] = max((hi - lo) / 2, 1e-6)
    uc = (edges[:-1] + edges[1:]) / 2; okb = half > 0
    midu = np.interp(u, uc[okb], mid[okb]); halfu = np.interp(u, uc[okb], half[okb])
    lat = vv - midu; r = np.clip(lat / halfu, -1.5, 1.5)
    # lateral tangent in 3D: direction of increasing |lat|, projected to the tangent plane
    tree = cKDTree(X); _, nb24 = tree.query(X, k=24)
    dX = X[nb24] - X[:, None]; dl = lat[nb24] - lat[:, None]
    g = np.einsum("nk,nkj->nj", dl, dX); g -= (g * n).sum(1, keepdims=True) * n
    tl = g / np.linalg.norm(g, axis=1, keepdims=True).clip(1e-12)
    tl *= np.sign(lat + 1e-12)[:, None]          # points away from the midrib on both halves
    assumptions = [dict(item="adaxial side", value="side facing the input camera", status="assumption",
                        check=f"{agree:.3f} of input-visible Gaussians face the camera after unpinned sign propagation (1 = consistently orientable)"),
                   dict(item="midrib", value="per-u-bin midpoint of the flattened lateral extent", status="approximation",
                        check=f"Isomap stress {stress:.3f} (1 - corr(geodesic, embedded distance))"),
                   dict(item="tip vs base", value="not inferred", status="unknown"),
                   dict(item="input camera fit", value=f"pose {cam['pose']}", status="measured", check=f"Umeyama residual {cam['fit_residual_rel']:.2e} L")]
    return dict(X=X, v=v, L=L, n=n, nb=nb, u=u, lat=lat, r=r, half=halfu, t_lat=tl, cam=cam, stress=stress, iso=iso,
                visible=cam["visible"], assumptions=assumptions, ply=ply)


def ear_frame(sid, ply=None, k_clusters=4):
    from scipy.cluster.vq import kmeans2
    ply = ply or f"{ROOT}/results/pilot/sam3d/{sid}/seed0/splat_ge.ply"; v = load_vertices(ply); X = xyz(v)
    L = float(np.linalg.norm(X.max(0) - X.min(0))); c = X.mean(0)
    ev, evec = np.linalg.eigh(np.cov((X - c).T)); ax = evec[:, -1]
    a = (X - c) @ ax; radial = (X - c) - a[:, None] * ax; n = radial / np.linalg.norm(radial, axis=1, keepdims=True).clip(1e-12)
    lab = srgb_to_lab(rgb(v)); cent, lbl = kmeans2(lab, k_clusters, seed=0, minit="++")
    chroma = np.hypot(cent[:, 1], cent[:, 2])
    kernel_c = int(np.argmax(cent[:, 2] + 0.2 * cent[:, 0]))                 # most yellow / bright cluster
    gall_c = int(np.argmin(chroma - 0.3 * cent[:, 0]))                       # least chroma, brightest: grey-white galls
    if gall_c == kernel_c: gall_c = -1
    kernel_ok = bool(cent[kernel_c, 2] >= 30.0)             # yellow kernels have b* ~ 45-50; below 30 no kernel tissue is identifiable
    cam = input_camera(sid, ply)
    assumptions = [dict(item="ear axis", value="first principal axis", status="approximation"),
                   dict(item="kernel tissue", value=f"Lab k-means cluster {kernel_c} (centre {np.round(cent[kernel_c], 1).tolist()})", status="inferred from photo colour"),
                   dict(item="existing galls", value=(f"cluster {gall_c} (centre {np.round(cent[gall_c], 1).tolist()})" if gall_c >= 0 else "not separated"), status="inferred from photo colour"),
                   dict(item="tip vs base", value="not inferred", status="unknown")]
    if not kernel_ok:
        assumptions[1] = dict(item="kernel tissue", value=f"NONE identifiable (most yellow cluster b* = {cent[kernel_c, 2]:.1f} < 30)", status="abstain")
    return dict(X=X, v=v, L=L, n=n, axis=ax, a=a, kernel=(lbl == kernel_c) if kernel_ok else np.zeros(len(X), bool), kernel_ok=kernel_ok, gall=(lbl == gall_c) if gall_c >= 0 else np.zeros(len(X), bool),
                lab=lab, clusters=cent.tolist(), cam=cam, visible=cam["visible"], assumptions=assumptions, ply=ply)
