"""
Render 24 views from leaf_splat.ply using SAM3D's renderer and write
COLMAP-format sparse/0/{cameras,images,points3D}.txt so GaussianEditor
can consume them.
"""
import os, sys, math
os.environ["LIDRA_SKIP_INIT"] = "true"
os.environ["CUDA_HOME"] = os.environ.get("CONDA_PREFIX", "")
sys.path.insert(0, "/work/mech-ai-scratch/UshB/sage-extend/sam-3d-objects/notebook")

import torch
import numpy as np
from PIL import Image
import utils3d
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

OUT_DIR   = "/work/mech-ai-scratch/UshB/sage-extend/leaf_colmap"
PLY_PATH  = "/work/mech-ai-scratch/UshB/sage-extend/leaf_output/leaf_splat.ply"
IMG_W, IMG_H = 512, 512
FOV_DEG   = 40
N_VIEWS   = 24
RADIUS    = 2.0
PITCH_DEG = 15   # slight downward angle so leaf is visible

os.makedirs(f"{OUT_DIR}/images",    exist_ok=True)
os.makedirs(f"{OUT_DIR}/sparse/0",  exist_ok=True)

# ── load gaussian splat ─────────────────────────────────────────────────────
from sam3d_objects.pipeline.gaussian.gaussian_model import GaussianModel
gs = GaussianModel(sh_degree=3)
gs.load_ply(PLY_PATH)
gs = gs.cuda()

# ── build camera poses ──────────────────────────────────────────────────────
fov = math.radians(FOV_DEG)
yaws   = [2 * math.pi * i / N_VIEWS - math.pi/2 for i in range(N_VIEWS)]
pitch  = math.radians(PITCH_DEG)

extrinsics, intrinsics = [], []
for yaw in yaws:
    orig = torch.tensor([
        math.sin(yaw) * math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw) * math.cos(pitch),
    ]).float().cuda() * RADIUS
    extr = utils3d.torch.extrinsics_look_at(
        orig,
        torch.zeros(3).float().cuda(),
        torch.tensor([0, 1, 0]).float().cuda(),
    )
    intr = utils3d.torch.intrinsics_from_fov_xy(
        torch.tensor(fov).cuda(),
        torch.tensor(fov).cuda(),
    )
    extrinsics.append(extr)
    intrinsics.append(intr)

# ── render frames ───────────────────────────────────────────────────────────
print(f"Rendering {N_VIEWS} views...")
frames = render_utils.render_frames(
    gs,
    extrinsics,
    intrinsics,
    {"resolution": IMG_W, "bg_color": (1, 1, 1), "backend": "gsplat"},
)
# frames: list of (H,W,3) uint8 numpy or tensor
for i, frame in enumerate(frames):
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    if frame.dtype != np.uint8:
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(frame).save(f"{OUT_DIR}/images/{i:04d}.png")
print(f"Saved {N_VIEWS} images to {OUT_DIR}/images/")

# ── write COLMAP cameras.txt (single PINHOLE camera shared by all views) ────
fx = fy = (IMG_W / 2) / math.tan(fov / 2)
cx, cy = IMG_W / 2, IMG_H / 2

with open(f"{OUT_DIR}/sparse/0/cameras.txt", "w") as f:
    f.write("# Camera list with one line of data per camera:\n")
    f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f.write(f"1 PINHOLE {IMG_W} {IMG_H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

# ── write COLMAP images.txt ──────────────────────────────────────────────────
# COLMAP format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
# extr from utils3d is a 4x4 world-to-camera matrix
with open(f"{OUT_DIR}/sparse/0/images.txt", "w") as f:
    f.write("# Image list with two lines of data per image:\n")
    f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    for i, extr in enumerate(extrinsics):
        w2c = extr.cpu().numpy()   # 4x4
        R = w2c[:3, :3]
        t = w2c[:3,  3]
        # rotation matrix → quaternion (w, x, y, z)
        tr = R[0,0] + R[1,1] + R[2,2]
        if tr > 0:
            s = 0.5 / math.sqrt(tr + 1.0)
            qw = 0.25 / s
            qx = (R[2,1] - R[1,2]) * s
            qy = (R[0,2] - R[2,0]) * s
            qz = (R[1,0] - R[0,1]) * s
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s
        f.write(f"{i+1} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} 1 {i:04d}.png\n")
        f.write("\n")  # empty POINTS2D line

# ── write empty points3D.txt ─────────────────────────────────────────────────
with open(f"{OUT_DIR}/sparse/0/points3D.txt", "w") as f:
    f.write("# 3D point list (empty — cameras only)\n")

print("COLMAP files written:")
print(f"  {OUT_DIR}/sparse/0/cameras.txt")
print(f"  {OUT_DIR}/sparse/0/images.txt")
print(f"  {OUT_DIR}/sparse/0/points3D.txt")
print("Done. Use --colmap_dir", OUT_DIR, "and --gs_source", PLY_PATH)
