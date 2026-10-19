"""E1 alignment study (A0-A3) for SAM 3D splats rendered at the input camera.

A0  existing pipeline: SAM 3D pose + intrinsics fitted to MoGe pointmap (src/rendering/input_view.py).
A1  camera-convention check: K residual on the pointmap; alternative K (principal point at centre).
A2  closed-form correction: object scale about its centroid (area match) + image-plane translation (centroid match),
    applied at the object's depth. Geometry untouched.
A3  render-and-compare of the 7-DoF object pose (rotation, translation, log-scale) with frozen geometry,
    loss = mask only (soft IoU + BCE). RGB and MoGe depth are NOT used -> held-out checks.
A4/A5 (geometry refinement) are not run unless A3 fails; see reports/01_camera_alignment_fixes.md.

Run in `sam3d` env.  usage: align.py <run_dir> <input_dir> <out_dir>
"""
import os, sys, json, time
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
sys.path.insert(0, os.path.expanduser("~/Desktop/Research/external/sam-3d-objects"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np, torch
from PIL import Image
from scipy import ndimage
from pytorch3d.transforms import quaternion_multiply, quaternion_invert, axis_angle_to_quaternion, quaternion_to_matrix
from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer
from gsplat import rasterization
from rendering.input_view import load_splat, fit_intrinsics, C0
from evaluation import metrics as M

DEV = "cuda"


def posed_gaussians(run_dir, ply=None):
    """Gaussians in OpenCV camera space (SAM 3D pose applied), exactly as input_view.py. `ply` may be any asset in
    the same object frame (edited assets: GaussianEditor and the geometry editor keep SAM 3D's object coordinates)."""
    s = load_splat(ply or os.path.join(run_dir, "splat.ply")); P = np.load(os.path.join(run_dir, "pose.npz"))
    s["op"] = np.nan_to_num(s["op"], posinf=15.94, neginf=-15.94)
    t = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(DEV)
    q, tr, sc = t(P["rotation"]).reshape(1, 4), t(P["translation"]).reshape(1, 3), t(P["scale"]).reshape(1, 3)
    means = SceneVisualizer.object_pointcloud(points_local=t(s["xyz"])[None], quat_l2c=q, trans_l2c=tr, scale_l2c=sc).points_list()[0]
    quats = quaternion_multiply(quaternion_invert(q), torch.nn.functional.normalize(t(s["rot"]), dim=-1))
    flip = torch.tensor([0.0, 0.0, 0.0, 1.0], device=DEV)
    G = dict(means=means * torch.tensor([-1.0, -1.0, 1.0], device=DEV), quats=quaternion_multiply(flip.expand_as(quats), quats),
             scales=torch.exp(t(s["scale"])) * sc[0, 0], op=torch.sigmoid(t(s["op"])), rgb=(0.5 + C0 * t(s["dc"])).clamp(0, 1))
    return G, P["pointmap"].astype(np.float32)


def transform(G, aa, tvec, logs, c):
    """Similarity about object centroid c: x' = s R (x - c) + c + t."""
    R = quaternion_to_matrix(axis_angle_to_quaternion(aa)); s = torch.exp(logs)
    return dict(G, means=(G["means"] - c) @ R.T * s + c + tvec, quats=quaternion_multiply(axis_angle_to_quaternion(aa).expand_as(G["quats"]), G["quats"]),
                scales=G["scales"] * s)


def render(G, K, W, H):
    rgb, a, _ = rasterization(G["means"], G["quats"], G["scales"], G["op"], G["rgb"], torch.eye(4, device=DEV)[None], K[None], W, H)
    return rgb[0] + (1 - a[0]), a[0, ..., 0]


def boundary_f(pred, gt, tol):
    bp = pred ^ ndimage.binary_erosion(pred); bg = gt ^ ndimage.binary_erosion(gt)
    if bp.sum() == 0 or bg.sum() == 0: return 0.0
    dg = ndimage.distance_transform_edt(~bg); dp = ndimage.distance_transform_edt(~bp)
    P = (dg[bp] <= tol).mean(); R = (dp[bg] <= tol).mean()
    return float(2 * P * R / (P + R + 1e-9))


def sil_metrics(alpha, mask):
    pred = alpha > 0.5; H, W = mask.shape; diag = np.hypot(H, W)
    out = dict(iou=M.mask_iou(pred, mask), bf=boundary_f(pred, mask, max(2.0, 0.0075 * diag)))
    if pred.sum() == 0: return dict(out, centroid_err=float("nan"), area_err=-1.0, bbox_iou=0.0)
    cp, cm = np.array(ndimage.center_of_mass(pred)), np.array(ndimage.center_of_mass(mask))
    bb = lambda m: (lambda y, x: (y.min(), x.min(), y.max(), x.max()))(*np.nonzero(m))
    a, b = bb(pred), bb(mask); iy = max(0, min(a[2], b[2]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iy * ix; ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return dict(out, centroid_err=float(np.linalg.norm(cp - cm) / diag), area_err=float(pred.sum() / mask.sum() - 1), bbox_iou=float(inter / max(ua, 1)))


def depth_check(G, pm, mask, K, W, H):
    """Held-out: median |z_obj - z_moge| / z_moge over mask pixels hit by the object's front surface (expected depth render)."""
    Gd = dict(G, rgb=G["means"][:, 2:3].expand(-1, 3).contiguous())
    zr, a, _ = rasterization(Gd["means"], Gd["quats"], Gd["scales"], Gd["op"], Gd["rgb"], torch.eye(4, device=DEV)[None], K[None], W, H)
    a = a[0, ..., 0].cpu().numpy(); z = (zr[0, ..., 0].cpu().numpy() / np.maximum(a, 1e-6))
    pmr = np.asarray(Image.fromarray(pm[..., 2]).resize((W, H), Image.NEAREST)) if pm.shape[:2] != (H, W) else pm[..., 2]
    sel = mask & (a > 0.5) & np.isfinite(pmr) & (pmr > 0)
    return float(np.median(np.abs(z[sel] - pmr[sel]) / pmr[sel])) if sel.sum() > 50 else float("nan")


def k_residual(pm, K):
    H, W, _ = pm.shape; v, u = np.mgrid[0:H, 0:W]; X, Y, Z = -pm[..., 0], -pm[..., 1], pm[..., 2]
    ok = np.isfinite(pm).all(-1) & (Z > 1e-4); pu = K[0, 0] * X[ok] / Z[ok] + K[0, 2]; pv = K[1, 1] * Y[ok] / Z[ok] + K[1, 2]
    return float(np.median(np.hypot(pu - (u[ok] + .5), pv - (v[ok] + .5))))


def main(run_dir, inp, out):
    os.makedirs(out, exist_ok=True); t0 = time.time()
    img = np.asarray(Image.open(os.path.join(inp, "image.png")).convert("RGB")); H, W = img.shape[:2]
    mask = np.asarray(Image.open(os.path.join(inp, "0.png")).convert("L")) > 127
    G, pm = posed_gaussians(run_dir)
    pm_cv = pm * np.array([-1, -1, 1], np.float32)          # pointmap stored in PyTorch3D cam; depth = +z either way
    K0 = fit_intrinsics(pm); kres = k_residual(pm, K0)
    K0[0] *= W / pm.shape[1]; K0[1] *= H / pm.shape[0]; K = torch.from_numpy(K0).to(DEV)
    ref = img.copy(); ref[~mask] = 255; res, imgs = {}, {}

    def score(name, Gx, Kx):
        with torch.no_grad():
            rgb, a = render(Gx, Kx, W, H)
        rgb = (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8); a = a.cpu().numpy(); both = (a > 0.5) & mask
        r = sil_metrics(a, mask); r.update(lpips=M.lpips_masked(rgb, ref, mask), delta_e=M.delta_e_in_mask(rgb, img, both) if both.sum() else float("nan"),
                                         depth_rel_err=depth_check(Gx, pm, mask, Kx, W, H))
        res[name] = r; imgs[name] = (rgb, a > 0.5); Image.fromarray(rgb).save(f"{out}/{name}.png")

    # A0
    score("A0", G, K)
    # A1: principal point at image centre, same focal (tests whether the fitted cx,cy are the error source)
    K1 = K.clone(); K1[0, 2], K1[1, 2] = W / 2, H / 2; score("A1_pp_centre", G, K1)
    # A2: closed form (2 fixed-point iterations), on A0 camera
    c = G["means"].mean(0); aa = torch.zeros(3, device=DEV); tv = torch.zeros(3, device=DEV); ls = torch.zeros((), device=DEV)
    for _ in range(3):
        with torch.no_grad():
            _, a = render(transform(G, aa, tv, ls, c), K, W, H)
        pred = a.cpu().numpy() > 0.5
        if pred.sum() < 10: break
        ls = ls + 0.5 * float(np.log(mask.sum() / pred.sum()))
        with torch.no_grad():
            _, a = render(transform(G, aa, tv, ls, c), K, W, H)
        pred = a.cpu().numpy() > 0.5; cy_p, cx_p = ndimage.center_of_mass(pred); cy_m, cx_m = ndimage.center_of_mass(mask)
        z = float((c + tv)[2]); tv = tv + torch.tensor([(cx_m - cx_p) * z / K0[0, 0], (cy_m - cy_p) * z / K0[1, 1], 0.0], device=DEV, dtype=torch.float32)
    score("A2_closed_form", transform(G, aa, tv, ls, c), K); a2 = (aa.clone(), tv.clone(), ls.clone())
    # A3: render-and-compare, mask loss only, from A2 init
    s = 512 / max(H, W); w, h = int(round(W * s)), int(round(H * s)); Ks = K.clone(); Ks[:2] *= s
    m_t = torch.from_numpy(np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)) / 255.0).float().to(DEV)
    aa_p, tv_p, ls_p = [torch.nn.Parameter(x.clone()) for x in a2]
    opt = torch.optim.Adam([{"params": [aa_p], "lr": 5e-3}, {"params": [tv_p], "lr": 2e-3 * float(c[2])}, {"params": [ls_p], "lr": 1e-2}])
    hist = []
    for it in range(300):
        _, a = render(transform(G, aa_p, tv_p, ls_p, c), Ks, w, h); a = a.clamp(1e-4, 1 - 1e-4)
        inter = (a * m_t).sum(); loss = 1 - inter / (a.sum() + m_t.sum() - inter) + 0.5 * torch.nn.functional.binary_cross_entropy(a, m_t)
        opt.zero_grad(); loss.backward(); opt.step(); hist.append(float(loss))
    with torch.no_grad():
        score("A3_render_compare", transform(G, aa_p, tv_p, ls_p, c), K)
    pose = dict(A2=dict(rot_axis_angle=a2[0].tolist(), t=a2[1].tolist(), log_scale=float(a2[2])),
                A3=dict(rot_axis_angle=aa_p.detach().tolist(), t=tv_p.detach().tolist(), log_scale=float(ls_p.detach()),
                        rot_deg=float(np.degrees(aa_p.detach().norm().cpu())), loss_first_last=[hist[0], hist[-1]]),
                centroid_obj_cam=c.tolist())
    # overlay: mask (blue) vs render silhouette (red) for A0 and A3
    for nm in ["A0", "A3_render_compare"]:
        o = img.astype(float) * 0.5 + 127; pr = imgs[nm][1]
        o[mask & ~pr] = [40, 90, 255]; o[pr & ~mask] = [255, 60, 40]; o[pr & mask] = 0.5 * o[pr & mask] + [0, 80, 0]
        Image.fromarray(o.clip(0, 255).astype(np.uint8)).save(f"{out}/overlay_{nm}.png")
    rep = dict(run_dir=run_dir, W=W, H=H, K=K0.tolist(), K_pointmap_reproj_px=kres, metrics=res, pose_correction=pose,
               mask_area_frac=float(mask.mean()), sec=time.time() - t0)
    json.dump(rep, open(f"{out}/alignment.json", "w"), indent=1)
    print(json.dumps({k: {m: round(v, 3) for m, v in r.items()} for k, r in res.items()}))


def render_at_input(ply, sid, root=None, corrected=True):
    """Render any asset of specimen `sid` at the input camera, with the A3* pose correction (if corrected)."""
    root = root or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_dir = f"{root}/results/pilot/sam3d/{sid}/seed0"; A = json.load(open(f"{root}/results/alignment/{sid}/alignment.json"))
    W, H = A["W"], A["H"]; K = torch.tensor(A["K"], dtype=torch.float32, device=DEV)
    G, _ = posed_gaussians(run_dir, ply); G0, _ = posed_gaussians(run_dir)
    if corrected:
        m = A["metrics"]; sel = "A3" if m["A3_render_compare"]["iou"] > m["A2_closed_form"]["iou"] else "A2"; pc = A["pose_correction"][sel]
        c = torch.tensor(A["pose_correction"]["centroid_obj_cam"], device=DEV)      # centroid of the ORIGINAL asset (same pivot as the fit)
        G = transform(G, torch.tensor(pc["rot_axis_angle"], device=DEV), torch.tensor(pc["t"], device=DEV), torch.tensor(pc["log_scale"], device=DEV), c)
    with torch.no_grad():
        rgb, a = render(G, K, W, H)
    return (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), a.cpu().numpy()


if __name__ == "__main__":
    main(*sys.argv[1:4])
