"""Shared camera rig + COLMAP writer + 3DGS rendering (GaussianEditor's own rasterizer).

Run inside the `gedit` env. The same rig is used for (a) GaussianEditor's training
views, (b) E4 multi-view consistency, and (c) figure renders, so every arm is
evaluated from identical cameras.

usage: views.py <splat.ply> <out_dir> [--res 512] [--n_az 12]
"""
import os, sys, json, math, argparse
import numpy as np, torch
from PIL import Image

GE = os.path.expanduser("~/Desktop/Research/external/GaussianEditor")
sys.path.insert(0, GE)
from gaussiansplatting.scene.gaussian_model import GaussianModel
from gaussiansplatting.scene.cameras import Simple_Camera
from gaussiansplatting.gaussian_renderer import render

FOV_DEG = 40.0


def rig(n_az=12, elevs=(20.0, -20.0), extra=((0.0, 80.0), (0.0, -80.0))):
    """(azimuth_deg, elevation_deg) list. Elevation <0 looks from below (underside)."""
    v = [(360.0 * i / n_az, e) for e in elevs for i in range(n_az)]
    return v + list(extra)


def look_at_w2c(eye, target, up=np.array([0, 1.0, 0])):
    """OpenCV/COLMAP convention: x right, y down, z forward."""
    z = target - eye; z /= np.linalg.norm(z)
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([0, 0, 1.0])
    x = np.cross(z, -up); x /= np.linalg.norm(x)   # right
    y = np.cross(z, x)                              # down
    R = np.stack([x, y, z])                         # rows = camera axes in world
    return R, -R @ eye


def rotmat2qvec(R):
    m = R; tr = np.trace(m)
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1); return np.array([0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]); return np.array([(m[2, 1] - m[1, 2]) / s, .25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]); return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, .25 * s, (m[1, 2] + m[2, 1]) / s])
    s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]); return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, .25 * s])


def load_gs(ply):
    g = GaussianModel(sh_degree=0, anchor_weight_init_g0=1.0, anchor_weight_init=0.1, anchor_weight_multiplier=2)
    g.load_ply(ply)
    return g


def cameras_for(g, res=512, n_az=12):
    xyz = g.get_xyz.detach().cpu().numpy()
    op = g.get_opacity.detach().cpu().numpy().ravel()
    keep = xyz[op > 0.05] if (op > 0.05).sum() > 100 else xyz
    c = np.median(keep, 0); r = np.percentile(np.linalg.norm(keep - c, axis=1), 98)
    dist = 1.15 * r / math.tan(math.radians(FOV_DEG) / 2)
    cams = []
    for k, (az, el) in enumerate(rig(n_az)):
        a, e = math.radians(az), math.radians(el)
        eye = c + dist * np.array([math.cos(e) * math.sin(a), -math.sin(e), math.cos(e) * math.cos(a)])  # -y is up in SAM3D frame? see note
        R, t = look_at_w2c(eye, c, up=np.array([0, -1.0, 0]))
        cams.append(dict(id=k, az=az, el=el, R=R, t=t, name=f"{k:03d}"))
    return cams, dict(center=c.tolist(), radius=float(r), dist=float(dist), fov_deg=FOV_DEG, res=res)


class _Pipe:
    convert_SHs_python = False; compute_cov3D_python = False; debug = False


def render_views(g, cams, res=512, bg=(1.0, 1.0, 1.0)):
    fov = math.radians(FOV_DEG); out = []
    bgt = torch.tensor(bg, dtype=torch.float32, device="cuda")
    for cam in cams:
        sc = Simple_Camera(cam["id"], cam["R"].T, cam["t"], fov, fov, res, res, cam["name"], cam["id"])
        with torch.no_grad():
            pkg = render(sc, g, _Pipe(), bgt)
        img = (pkg["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out.append(img)
    return out


def write_colmap(out_dir, cams, imgs, res):
    os.makedirs(f"{out_dir}/images", exist_ok=True); os.makedirs(f"{out_dir}/sparse/0", exist_ok=True)
    f = (res / 2) / math.tan(math.radians(FOV_DEG) / 2)
    open(f"{out_dir}/sparse/0/cameras.txt", "w").write(f"1 PINHOLE {res} {res} {f:.6f} {f:.6f} {res/2} {res/2}\n")
    with open(f"{out_dir}/sparse/0/images.txt", "w") as fh:
        for cam, im in zip(cams, imgs):
            Image.fromarray(im).save(f"{out_dir}/images/{cam['name']}.png")
            q = rotmat2qvec(cam["R"]); t = cam["t"]
            fh.write(f"{cam['id']+1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {cam['name']}.png\n\n")
    open(f"{out_dir}/sparse/0/points3D.txt", "w").write("")


def write_points_ply(out_dir, g):
    """GaussianEditor's COLMAP reader needs a point cloud; use the splat's own centres + DC colours."""
    from plyfile import PlyData, PlyElement
    xyz = g.get_xyz.detach().cpu().numpy()
    rgb = (np.clip(0.5 + 0.28209479177387814 * g._features_dc.detach().cpu().numpy()[:, 0, :], 0, 1) * 255).astype(np.uint8)
    dt = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    el = np.empty(len(xyz), dtype=dt); el["x"], el["y"], el["z"] = xyz.T; el["nx"] = el["ny"] = el["nz"] = 0
    el["red"], el["green"], el["blue"] = rgb.T
    PlyData([PlyElement.describe(el, "vertex")]).write(f"{out_dir}/sparse/0/points3D.ply")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ply"); ap.add_argument("out"); ap.add_argument("--res", type=int, default=512); ap.add_argument("--n_az", type=int, default=12)
    ap.add_argument("--rig", default=None, help="reuse cameras from an existing rig.json (identical views across arms)")
    a = ap.parse_args()
    g = load_gs(a.ply)
    if a.rig:
        meta = json.load(open(a.rig)); meta.pop("views_from", None)
        cams = [dict(id=v["id"], az=v["az"], el=v["el"], R=np.array(v["R"]), t=np.array(v["t"]), name=f"{v['id']:03d}") for v in meta["views"]]
        meta["views_from"] = a.rig
    else:
        cams, meta = cameras_for(g, a.res, a.n_az)
    imgs = render_views(g, cams, meta.get("res", a.res))
    write_colmap(a.out, cams, imgs, a.res); write_points_ply(a.out, g)
    meta["views"] = [dict(id=c["id"], az=c["az"], el=c["el"], R=c["R"].tolist(), t=c["t"].tolist()) for c in cams]
    json.dump(meta, open(f"{a.out}/rig.json", "w"), indent=1)
    print("rendered", len(imgs), "views ->", a.out)
