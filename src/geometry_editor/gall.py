"""Volumetric gall editing (v2): new closed, lobed gall bodies made of new Gaussians, replacing kernel tissue.

Why: the v1 `outgrowth` operator displaces the existing surface along its normal (a height field). A height field
cannot (i) add a closed volume with an underside, (ii) overhang neighbouring tissue (undercut), or (iii) remove the
tissue a gall replaces. Corn smut galls are bulbous masses that replace kernels (UNL CropWatch), so v2 builds them
explicitly:
  * body = union of m spheres (m = 3-6, radii 0.35-0.6 D, centres around c + 0.3 D n with tangential jitter):
    a lobed mass of diameter ~D attached at the kernel surface point c with outward normal n;
  * surface Gaussians are sampled on the union boundary (points of each sphere outside all others), oriented to
    the local sphere normal (thin axis along it), spacing ~ the median nearest-neighbour spacing of the organ;
  * original Gaussians inside the body are made transparent (opacity logit -> -16): the gall REPLACES that tissue;
  * colour: sampled from the real photo's own gall tissue if the ear frame found it (grey-white cluster), else the
    source's words "white to gray" (Lab anchors, assumption). Lighting/shading is not modelled.
Outputs <out>.ply (original N Gaussians first, then the gall Gaussians) and <out>.json with per-gall geometry:
volume (Monte-Carlo on the sphere union), equivalent diameter, undercut fraction, kernels replaced.
"""
import json
import numpy as np
from scipy.spatial import cKDTree
from plyfile import PlyData, PlyElement
from .anatomy import rgb, rgb_to_dc
from .bilayer import lab_to_srgb, mat_to_quat


def lobes(c, n, D, rng):
    m = int(rng.integers(7, 13)); t1 = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0]); t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)
    C, R = [], []
    for _ in range(m):
        ang = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0, 0.3 * D)
        C.append(c + 0.25 * D * n + rr * (np.cos(ang) * t1 + np.sin(ang) * t2) + rng.uniform(-0.1, 0.2) * D * n); R.append(rng.uniform(0.25, 0.45) * D)
    R = np.array(R)
    return np.array(C), R


def inside(P, C, R):
    return (np.linalg.norm(P[:, None] - C[None], axis=2) < R[None]).any(1)


def build(E, centres, normals, diams, out_ply, rng=None, gall_rgb=None):
    rng = rng or np.random.default_rng(0); v = E["v"]; X = E["X"]; N = len(X)
    sp = float(np.median(cKDTree(X).query(X[rng.choice(N, min(5000, N), replace=False)], k=2)[0][:, 1]))
    real_gall = E["gall"].any()
    pal = rgb(v)[E["gall"]] if real_gall else lab_to_srgb(np.array([[80.0, 0, 3], [62.0, 0, 2]]))
    new = []; galls = []; hide = np.zeros(N, bool); organ_c = X.mean(0)
    for c, n, D in zip(centres, normals, diams):
        C, R = lobes(c, n, D, rng)
        pts, nrm = [], []
        for cc, rr in zip(C, R):
            k = max(200, int(4 * np.pi * rr * rr / (sp * sp)))
            u = rng.normal(size=(k, 3)); u /= np.linalg.norm(u, axis=1, keepdims=True); p = cc + rr * u
            others = [j for j in range(len(C)) if not np.allclose(C[j], cc)]
            keep = ~inside(p, C[others], R[others] * 0.999) if others else np.ones(k, bool)
            pts.append(p[keep]); nrm.append(u[keep])
        P = np.concatenate(pts); Nn = np.concatenate(nrm)
        # drop gall surface points buried in the ear (behind the attachment plane)
        keep = ((P - c) @ n) > -0.15 * D; P, Nn = P[keep], Nn[keep]
        hid = inside(X, C, R); hide |= hid
        # Monte-Carlo volume of the union (outside the ear half-space at the attachment)
        box_lo, box_hi = (C - R[:, None]).min(0), (C + R[:, None]).max(0); Q = rng.uniform(box_lo, box_hi, size=(60000, 3))
        inq = inside(Q, C, R) & (((Q - c) @ n) > -0.15 * D); vol = float(inq.mean() * np.prod(box_hi - box_lo))
        radial = (P - organ_c); radial /= np.linalg.norm(radial, axis=1, keepdims=True)
        undercut = float(((Nn * n).sum(1) < -0.2).mean())          # surface facing back toward the organ
        galls.append(dict(centre=c.tolist(), diameter=float(D), n_lobes=len(C), volume=vol, eq_diameter=float((6 * vol / np.pi) ** (1 / 3)),
                          undercut_frac=undercut, kernels_replaced=int((hid & E["kernel"]).sum()), n_gaussians=int(len(P))))
        new.append((P, Nn))
    P = np.concatenate([p for p, _ in new]); Nn = np.concatenate([q for _, q in new]); M = len(P)
    e1 = np.cross(Nn, np.where(np.abs(Nn[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    q = mat_to_quat(np.stack([e1, np.cross(Nn, e1), Nn], 2))
    # albedo: median of the real gall tissue (or KB words) with small per-Gaussian variation; then BAKED shading,
    # because SAM 3D splats carry baked lighting and unshaded spheres read as flat discs: Lambertian term with the light
    # on the input-camera side (as in the photo) + darkening toward the attachment crevice. Shading is a rendering
    # choice, not a biological claim.
    alb = np.median(pal, 0)[None] * (1 + rng.normal(0, 0.04, (M, 1)))
    l = E["cam"]["cam_obj"] - organ_c; l /= np.linalg.norm(l) + 1e-12
    att = np.concatenate([np.full(len(p_), 0.0) + ((p_ - c) @ n) / (0.5 * D) for (p_, _), c, n, D in zip(new, centres, normals, diams)])
    shade = (0.38 + 0.62 * np.clip(Nn @ l, 0, 1)) * (0.65 + 0.35 * np.clip(att, 0, 1))[:, None].ravel()
    col = np.clip(alb * (shade / np.median(shade))[:, None], 0, 1) if gall_rgb is None else gall_rgb(P)   # median shade = palette median
    g = np.zeros(M, dtype=v.dtype)
    g["x"], g["y"], g["z"] = P.T.astype(np.float32)
    for k in range(4): g[f"rot_{k}"] = q[:, k].astype(np.float32)
    for k, s in enumerate([sp * 1.2, sp * 1.2, sp * 0.3]): g[f"scale_{k}"] = np.float32(np.log(s))
    dc = rgb_to_dc(col)
    for k in range(3): g[f"f_dc_{k}"] = dc[:, k].astype(np.float32)
    g["opacity"] = np.float32(8.0)
    for k, nn in zip(("nx", "ny", "nz"), Nn.T):
        if k in g.dtype.names: g[k] = nn.astype(np.float32)
    w = v.copy(); w["opacity"][hide] = np.float32(-16.0)
    PlyData([PlyElement.describe(np.concatenate([w, g]), "vertex")], text=False).write(out_ply)
    rep = dict(n_original=N, n_added=M, n_hidden=int(hide.sum()), spacing=sp, galls=galls,
               colour_source="real gall tissue in this photo" if real_gall else "KB words 'white to gray' (assumption)")
    json.dump(rep, open(out_ply + ".json", "w"), indent=1); return rep
