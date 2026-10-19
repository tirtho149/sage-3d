"""E1: render a SAM 3D splat from its own input camera and score it against the real photo.

Run in the `sam3d` env. Pose handling follows SAM 3D's notebook make_scene():
points -> SceneVisualizer.object_pointcloud(l2c quat/trans/scale); gaussian quats
q' = q_l2c^-1 * q; scales *= s. Intrinsics are fitted by least squares to MoGe's
camera-space pointmap (PyTorch3D camera: +x left, +y up, +z forward).

usage: input_view.py <run_dir> <input_dir>   -> <run_dir>/input_view.png, e1.json
"""
import os, sys, json
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
sys.path.insert(0, os.path.expanduser("~/Desktop/Research/external/sam-3d-objects"))
import numpy as np, torch
from PIL import Image
from plyfile import PlyData
from pytorch3d.transforms import quaternion_multiply, quaternion_invert
from sam3d_objects.utils.visualization.scene_visualizer import SceneVisualizer
from gsplat import rasterization

C0 = 0.28209479177387814


def load_splat(ply):
    v = PlyData.read(ply)["vertex"].data
    g = lambda *n: np.stack([v[k] for k in n], 1).astype(np.float32)
    return dict(xyz=g("x", "y", "z"), dc=g("f_dc_0", "f_dc_1", "f_dc_2"), op=v["opacity"].astype(np.float32),
                scale=g("scale_0", "scale_1", "scale_2"), rot=g("rot_0", "rot_1", "rot_2", "rot_3"))


def fit_intrinsics(pm):
    """pm: HxWx3 camera-space points (PyTorch3D). Returns OpenCV K for the pointmap grid."""
    H, W, _ = pm.shape; v, u = np.mgrid[0:H, 0:W]
    X, Y, Z = -pm[..., 0], -pm[..., 1], pm[..., 2]          # to OpenCV (x right, y down)
    ok = np.isfinite(pm).all(-1) & (Z > 1e-4)
    A = lambda t: np.stack([t[ok], np.ones(ok.sum())], 1)
    fx, cx = np.linalg.lstsq(A(X / np.where(ok, Z, 1)), u[ok] + 0.5, rcond=None)[0]
    fy, cy = np.linalg.lstsq(A(Y / np.where(ok, Z, 1)), v[ok] + 0.5, rcond=None)[0]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)


def render_input_view(run_dir, W, H):
    s = load_splat(os.path.join(run_dir, "splat.ply")); P = np.load(os.path.join(run_dir, "pose.npz"))
    dev = "cuda"; t = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(dev)
    q, tr, sc = t(P["rotation"]).reshape(1, 4), t(P["translation"]).reshape(1, 3), t(P["scale"]).reshape(1, 3)
    pc = SceneVisualizer.object_pointcloud(points_local=t(s["xyz"])[None], quat_l2c=q, trans_l2c=tr, scale_l2c=sc)
    means = pc.points_list()[0]
    quats = quaternion_multiply(quaternion_invert(q), torch.nn.functional.normalize(t(s["rot"]), dim=-1))
    scales = torch.exp(t(s["scale"])) * sc[0, 0]
    means_cv = means * torch.tensor([-1.0, -1.0, 1.0], device=dev)        # PyTorch3D cam -> OpenCV cam
    flip = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)                  # 180deg about z: (w,x,y,z)
    quats_cv = quaternion_multiply(flip.expand_as(quats), quats)
    pm = P["pointmap"].astype(np.float32); K = fit_intrinsics(pm)
    K[0] *= W / pm.shape[1]; K[1] *= H / pm.shape[0]
    colors = (0.5 + C0 * t(s["dc"])).clamp(0, 1); op = torch.sigmoid(t(s["op"]))
    rgb, alpha, _ = rasterization(means_cv, quats_cv, scales, op, colors, torch.eye(4, device=dev)[None], t(K)[None], W, H)
    rgb = rgb[0] + (1 - alpha[0])        # composite on white
    return (rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), alpha[0, ..., 0].cpu().numpy(), K


if __name__ == "__main__":
    run_dir, inp = sys.argv[1], sys.argv[2]
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from evaluation import metrics as M
    img = np.asarray(Image.open(os.path.join(inp, "image.png")).convert("RGB")); H, W = img.shape[:2]
    mask = np.asarray(Image.open(os.path.join(inp, "0.png")).convert("L")) > 127
    rgb, alpha, K = render_input_view(run_dir, W, H)
    Image.fromarray(rgb).save(os.path.join(run_dir, "input_view.png"))
    pred = alpha > 0.5; both = pred & mask
    ref = img.copy(); ref[~mask] = 255
    r = dict(mask_iou=M.mask_iou(pred, mask), delta_e=M.delta_e_in_mask(rgb, img, both), ssim=M.ssim_in_mask(rgb, img, both),
             lpips=M.lpips_masked(rgb, ref, mask), K=K.tolist(), pred_cov=float(pred.mean()), mask_cov=float(mask.mean()))
    json.dump(r, open(os.path.join(run_dir, "e1.json"), "w"), indent=1); print(json.dumps({k: v for k, v in r.items() if k != "K"}))
