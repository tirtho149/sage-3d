"""Correspondence-based cross-view consistency (brief s14). For adjacent ring views (i, i+1), project surface points
(Gaussian centres with opacity > 0.5) into both views, keep points visible in both (depth test against the rendered
depth, tolerance 1 % of the rig radius), and compare the EDITED colours at the two pixels (CIELAB dE). Reports the
median dE over correspondences; the unedited asset D gives the resampling floor. Internal consistency only.
  2D per-view edits (Bmv/Cmv): images = edits of D's renders; geometry = D asset.
  3D edits: images = renders of the edited asset; geometry = that asset.
usage (sam3d env): crossview.py <sid>  -> results/statistics/crossview_corr_<sid>.json
"""
import os, sys, json, math
import numpy as np
from PIL import Image
from plyfile import PlyData
from skimage.color import rgb2lab
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, f"{R}/src")
from rendering.rig_render import load_view


def pts(ply, n=30000, seed=0):
    v = PlyData.read(ply)["vertex"].data; op = 1 / (1 + np.exp(-np.nan_to_num(v["opacity"].astype(np.float64), posinf=15.9, neginf=-15.9)))
    X = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)[op > 0.5]
    return X[np.random.default_rng(seed).choice(len(X), min(n, len(X)), replace=False)]


def project(X, cam, rig):
    f = rig["res"] / 2 / math.tan(math.radians(rig["fov_deg"]) / 2); Xc = X @ np.array(cam["R"]).T + np.array(cam["t"])
    return f * Xc[:, 0] / Xc[:, 2] + rig["res"] / 2, f * Xc[:, 1] / Xc[:, 2] + rig["res"] / 2, Xc[:, 2]


def visible(u, v, z, depth, alpha, tol):
    H, W = depth.shape; ui, vi = np.round(u - 0.5).astype(int), np.round(v - 0.5).astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H); m = np.zeros(len(u), bool)
    d = depth[vi[ok], ui[ok]] / np.maximum(alpha[vi[ok], ui[ok]], 1e-3)
    m[np.nonzero(ok)[0]] = (alpha[vi[ok], ui[ok]] > 0.5) & (np.abs(d - z[ok]) < tol); return m, ui, vi


def score(X, imgs, depths, alphas, rig):
    cams = rig["views"][:8]; tol = 0.01 * rig["radius"] * 2; out = []
    proj = [project(X, c, rig) for c in cams]; vis = [visible(*p, depths[i], alphas[i], tol) for i, p in enumerate(proj)]
    for i in range(8):
        j = (i + 1) % 8; m = vis[i][0] & vis[j][0]
        if m.sum() < 50: continue
        a = rgb2lab(imgs[i][vis[i][2][m], vis[i][1][m]] / 255.0); b = rgb2lab(imgs[j][vis[j][2][m], vis[j][1][m]] / 255.0)
        out.append(float(np.median(np.linalg.norm(a - b, axis=1))))
    return dict(median_dE_adjacent=float(np.median(out)) if out else None, per_pair=out)


def main(sid):
    P = f"{R}/results/phaseA/{sid}"; rig = json.load(open(f"{R}/results/pilot/{sid}/D/rig.json")); res = {}
    Dv = [load_view(f"{P}/D_render", i) for i in range(8)]; XD = pts(f"{R}/results/pilot/sam3d/{sid}/seed0/splat_ge.ply")
    res["D (floor)"] = score(XD, [v[0].astype(float) for v in Dv], [v[2] for v in Dv], [v[1] for v in Dv], rig)
    for arm in ("Bmv", "Cmv"):
        d = f"{R}/results/pilot/{sid}/{arm}_masked"
        if all(os.path.exists(f"{d}/{i:03d}.png") for i in range(8)):
            ims = [np.asarray(Image.open(f"{d}/{i:03d}.png").convert("RGB").resize((rig["res"], rig["res"])), float) for i in range(8)]
            res[arm] = score(XD, ims, [v[2] for v in Dv], [v[1] for v in Dv], rig)
    for arm in ("E", "F", "K", "J", "G", "H"):
        d = f"{P}/{arm}/seed0"
        if os.path.exists(f"{d}/render/rgb/007.png") and os.path.exists(f"{d}/asset.ply"):
            V = [load_view(f"{d}/render", i) for i in range(8)]
            res[arm] = score(pts(f"{d}/asset.ply"), [v[0].astype(float) for v in V], [v[2] for v in V], [v[1] for v in V], rig)
    os.makedirs(f"{R}/results/statistics", exist_ok=True); json.dump(res, open(f"{R}/results/statistics/crossview_corr_{sid}.json", "w"), indent=1)
    return {k: v["median_dE_adjacent"] for k, v in res.items()}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1])))
