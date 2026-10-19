"""Figure 5 - Displacement analysis: per-Gaussian displacement heatmaps (fraction of bbox diagonal) and the
geometry-only / appearance-only decomposition of each 3D edit, on fixed rig views. Uses prune-aware matching."""
import os, sys, json
import numpy as np, matplotlib.pyplot as plt
from matplotlib import cm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures.common import ROOT, TW, show, save, crop_white
from metrics.gaussian_diff import load as gload, match
from rendering.rig_render import render_rig
from plyfile import PlyData

C0 = 0.28209479177387814; VMAX = 0.05


def arrays(p):
    v = PlyData.read(p)["vertex"].data; g = lambda *n: np.stack([v[k] for k in n], 1).astype(np.float32)
    q = g("rot_0", "rot_1", "rot_2", "rot_3"); q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = np.nan_to_num(v["opacity"].astype(np.float32), posinf=15.94, neginf=-15.94)
    return dict(means=g("x", "y", "z"), quats=q, scales=np.exp(g("scale_0", "scale_1", "scale_2")), op=1 / (1 + np.exp(-op)),
                rgb=np.clip(0.5 + C0 * g("f_dc_0", "f_dc_1", "f_dc_2"), 0, 1))


def main(sid="Peach__Leaf_Curl__0087", arms=("E", "J", "G", "F", "K", "H"), views=(0, 4, 17)):
    base = f"{ROOT}/results/phaseA/{sid}"; rig = json.load(open(f"{ROOT}/results/pilot/{sid}/D/rig.json"))
    src = f"{ROOT}/results/pilot/sam3d/{sid}/seed0/splat_ge.ply"; A = arrays(src); Ad = gload(src); L = float(np.linalg.norm(A["means"].max(0) - A["means"].min(0)))
    avail = [a for a in arms if os.path.exists(f"{base}/{a}/seed0/asset.ply")]
    fig, axs = plt.subplots(len(avail), 3 * len(views), figsize=(TW, TW * len(avail) / (3 * len(views)) * 1.05), squeeze=False)
    rows = {}
    for r, arm in enumerate(avail):
        p = f"{base}/{arm}/seed0/asset.ply"; B = arrays(p); ia, jb, _ = match(Ad, gload(p))
        d = np.linalg.norm(B["means"][jb] - A["means"][ia], axis=1) / L
        heat = np.full((len(B["means"]), 3), 0.6, np.float32); heat[jb] = cm.magma(np.clip(d / VMAX, 0, 1))[:, :3]
        geo = {k: B[k][jb] for k in B}; geo["rgb"] = A["rgb"][ia]                       # edited geometry, original colour
        app = {k: A[k][ia] for k in A}; app["rgb"] = B["rgb"][jb]                       # original geometry, edited colour
        R_heat = render_rig(p, rig, colors=heat, views=views); R_geo = render_rig(None, rig, G=geo, views=views); R_app = render_rig(None, rig, G=app, views=views)
        for c, (rh, rg, ra) in enumerate(zip(R_heat, R_geo, R_app)):
            show(axs[r][3 * c], crop_white(rh["rgb"]), f"{arm} disp. v{rh['id']}" if r == 0 else f"{arm}")
            show(axs[r][3 * c + 1], crop_white(rg["rgb"]), "geometry only" if r == 0 else None)
            show(axs[r][3 * c + 2], crop_white(ra["rgb"]), "appearance only" if r == 0 else None)
        rows[arm] = dict(median=float(np.median(d)), p95=float(np.percentile(d, 95)), n_matched=int(len(jb)))
    sm = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, VMAX * 100)); cax = fig.add_axes([0.92, 0.3, 0.008, 0.4])
    cb = fig.colorbar(sm, cax=cax); cb.set_label("displacement (% of bbox diagonal; unit-free)", fontsize=5); cb.ax.tick_params(labelsize=4.5)
    fig.subplots_adjust(right=0.9)
    save(fig, f"fig5_displacement_{sid}", [f"results/phaseA/{sid}/*/seed0/asset.ply", src],
         "Per-Gaussian displacement (prune-aware index matching) and decomposition of each 3D edit into a geometry-only render "
         "(edited positions/rotations/scales, original colours) and an appearance-only render (original geometry, edited colours). "
         f"Displacement medians/p95 (fraction of L): {json.dumps({k: [round(v['median'], 4), round(v['p95'], 4)] for k, v in rows.items()})}. "
         "All views are synthetic renders of synthetic assets.")
    return rows


if __name__ == "__main__":
    print(main(*sys.argv[1:2]))
