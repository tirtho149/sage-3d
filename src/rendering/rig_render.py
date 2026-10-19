"""Render a splat from the fixed per-asset camera rig (rig.json from views.py) with gsplat: RGB (white bg), alpha, depth.
Used for silhouette / depth / multi-view metrics so every arm is rendered by the same renderer and cameras.
usage (sam3d env): rig_render.py <splat.ply> <rig.json> <out_dir> [--res 512]  -> out_dir/{rgb,alpha,depth}/NNN.png|npy
"""
import os, sys, json, math, argparse
import numpy as np, torch
from PIL import Image
from plyfile import PlyData
from gsplat import rasterization

C0 = 0.28209479177387814


def load(p, dev):
    v = PlyData.read(p)["vertex"].data; g = lambda *n: torch.tensor(np.stack([v[k] for k in n], 1), dtype=torch.float32, device=dev)
    op = torch.tensor(v["opacity"], dtype=torch.float32, device=dev).nan_to_num(posinf=15.94, neginf=-15.94)
    return dict(means=g("x", "y", "z"), quats=torch.nn.functional.normalize(g("rot_0", "rot_1", "rot_2", "rot_3"), dim=-1),
                scales=torch.exp(g("scale_0", "scale_1", "scale_2")), op=torch.sigmoid(op), rgb=(0.5 + C0 * g("f_dc_0", "f_dc_1", "f_dc_2")).clamp(0, 1))


def render_rig(ply, rig, res=None, dev="cuda", colors=None, views=None, G=None):
    """colors: optional (N,3) array in [0,1] overriding per-Gaussian colour (e.g. displacement heatmaps).
    G: optional in-memory Gaussians (dict of numpy arrays: means, quats(wxyz), scales(linear), op(0-1), rgb) instead of a ply."""
    if G is None: G = load(ply, dev)
    else: G = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev) for k, v in G.items()}
    res = res or rig["res"]
    if colors is not None: G["rgb"] = torch.as_tensor(colors, dtype=torch.float32, device=dev)
    f = res / 2 / math.tan(math.radians(rig["fov_deg"]) / 2)
    K = torch.tensor([[f, 0, res / 2], [0, f, res / 2], [0, 0, 1]], dtype=torch.float32, device=dev)
    out = []
    for cam in rig["views"]:
        if views is not None and cam["id"] not in views: continue
        V = torch.eye(4, device=dev); V[:3, :3] = torch.tensor(cam["R"], device=dev); V[:3, 3] = torch.tensor(cam["t"], device=dev)
        rgbd, a, _ = rasterization(G["means"], G["quats"], G["scales"], G["op"], G["rgb"], V[None], K[None], res, res, render_mode="RGB+ED")
        rgb = rgbd[0, ..., :3] + (1 - a[0]); out.append(dict(id=cam["id"], rgb=(rgb.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
                                                                alpha=a[0, ..., 0].cpu().numpy(), depth=rgbd[0, ..., 3].cpu().numpy()))
    return out


def save_view(d, r):
    """rgb PNG; alpha 8-bit PNG; depth float16 compressed npz (disk: ~18 MB -> ~1 MB per 18-view set)."""
    i = r["id"]; Image.fromarray(r["rgb"]).save(f"{d}/rgb/{i:03d}.png")
    Image.fromarray((np.clip(r["alpha"], 0, 1) * 255).round().astype(np.uint8)).save(f"{d}/alpha/{i:03d}.png")
    np.savez_compressed(f"{d}/depth/{i:03d}.npz", d=r["depth"].astype(np.float16))


def load_view(d, i):
    """Returns rgb uint8, alpha float32 [0,1], depth float32 (accepts the old .npy format too)."""
    import os
    rgb = np.asarray(Image.open(f"{d}/rgb/{i:03d}.png").convert("RGB"))
    a = (np.asarray(Image.open(f"{d}/alpha/{i:03d}.png")).astype(np.float32) / 255.0) if os.path.exists(f"{d}/alpha/{i:03d}.png") else np.load(f"{d}/alpha/{i:03d}.npy").astype(np.float32)
    z = np.load(f"{d}/depth/{i:03d}.npz")["d"].astype(np.float32) if os.path.exists(f"{d}/depth/{i:03d}.npz") else np.load(f"{d}/depth/{i:03d}.npy").astype(np.float32)
    return rgb, a, z


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("ply"); ap.add_argument("rig"); ap.add_argument("out"); ap.add_argument("--res", type=int, default=None)
    a = ap.parse_args(); rig = json.load(open(a.rig))
    for k in ("rgb", "alpha", "depth"): os.makedirs(f"{a.out}/{k}", exist_ok=True)
    for r in render_rig(a.ply, rig, a.res):
        save_view(a.out, r)
    print("rendered", len(rig["views"]), "views ->", a.out)
