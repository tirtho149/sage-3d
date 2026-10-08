"""Render leaf_yellow_holes.ply to PNG using orthographic Gaussian splatting."""
import numpy as np
import sys, os
from PIL import Image

PLY_IN  = "/work/mech-ai-scratch/UshB/sage-extend/Modified/leaf_yellow_holes.ply"
OUT_DIR = "/work/mech-ai-scratch/UshB/sage-extend/Modified"

# ---------- load PLY -------------------------------------------------------
with open(PLY_IN, "rb") as f:
    while True:
        if f.readline().strip() == b"end_header":
            break
    raw = f.read()

PROPS = ["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity",
         "scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
N = 37120
arr = np.frombuffer(raw, dtype=np.dtype([(p, np.float32) for p in PROPS]), count=N).copy()

C0 = 0.28209479177387814

pos   = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)          # (N,3)
color = np.clip(np.stack([
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]
], axis=1) * C0 + 0.5, 0.0, 1.0)                                  # (N,3) RGB

opa   = (1.0 / (1.0 + np.exp(-arr["opacity"].astype(np.float64)))).astype(np.float32)

scales_exp = np.exp(np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], axis=1))

quats = np.stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]], axis=1)
quats = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8)

# ---------- vectorised 2D covariance (orthographic projection onto XY) -----
w, qx, qy, qz = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
# Build rotation matrices (N,3,3)
R = np.stack([
    1-2*(qy*qy+qz*qz), 2*(qx*qy-w*qz),  2*(qx*qz+w*qy),
    2*(qx*qy+w*qz),  1-2*(qx*qx+qz*qz),  2*(qy*qz-w*qx),
    2*(qx*qz-w*qy),  2*(qy*qz+w*qx),   1-2*(qx*qx+qy*qy)
], axis=1).reshape(N, 3, 3)

M = R * scales_exp[:, np.newaxis, :]        # M[i,j,k] = R[i,j,k]*s[i,k]
cov_00 = (M[:, 0, :] ** 2).sum(axis=1)     # XX component in world units
cov_11 = (M[:, 1, :] ** 2).sum(axis=1)     # YY
cov_01 = (M[:, 0, :] * M[:, 1, :]).sum(axis=1)  # XY

# ---------- canvas setup ---------------------------------------------------
X_MIN, X_MAX = pos[:, 0].min() - 0.01, pos[:, 0].max() + 0.01
Y_MIN, Y_MAX = pos[:, 1].min() - 0.01, pos[:, 1].max() + 0.01

IMG_W = 700
IMG_H = int(IMG_W * (Y_MAX - Y_MIN) / (X_MAX - X_MIN))
PX = IMG_W / (X_MAX - X_MIN)   # pixels per world unit, X
PY = IMG_H / (Y_MAX - Y_MIN)   # pixels per world unit, Y

# Screen-space covariance
sc00 = cov_00 * PX * PX
sc11 = cov_11 * PY * PY
sc01 = cov_01 * PX * PY

# Projected screen centres
cx = ((pos[:, 0] - X_MIN) * PX).astype(np.float32)
cy = ((Y_MAX - pos[:, 1]) * PY).astype(np.float32)   # flip Y

# Sort back-to-front (increasing Z = farther from viewer who looks in -Z)
order = np.argsort(pos[:, 2])

# RGBA canvas (float32, 0-1)
canvas = np.ones((IMG_H, IMG_W, 4), dtype=np.float32)
canvas[:, :, :3] = 1.0   # white background

SIGMA_CLIP = 3.0
MIN_ALPHA  = 0.005

print(f"Canvas: {IMG_W}×{IMG_H}  |  Gaussians: {N}")
print("Rasterising...", flush=True)

for step, i in enumerate(order):
    alpha_i = float(opa[i])
    if alpha_i < MIN_ALPHA:
        continue

    # Bounding box
    det  = sc00[i] * sc11[i] - sc01[i] ** 2 + 1e-8
    lam1 = 0.5 * ((sc00[i] + sc11[i]) + np.sqrt((sc00[i] - sc11[i])**2 + 4*sc01[i]**2 + 1e-8))
    radius = SIGMA_CLIP * np.sqrt(abs(lam1))

    x0 = max(0, int(cx[i] - radius))
    x1 = min(IMG_W, int(cx[i] + radius) + 1)
    y0 = max(0, int(cy[i] - radius))
    y1 = min(IMG_H, int(cy[i] + radius) + 1)
    if x0 >= x1 or y0 >= y1:
        continue

    # Pixel offsets from centre
    dx = (np.arange(x0, x1, dtype=np.float32) - cx[i])
    dy = (np.arange(y0, y1, dtype=np.float32) - cy[i])
    DX, DY = np.meshgrid(dx, dy)            # (h, w)

    # Mahalanobis distance via 2x2 inverse
    inv00 =  sc11[i] / det
    inv11 =  sc00[i] / det
    inv01 = -sc01[i] / det

    power = -0.5 * (DX*DX*inv00 + 2*DX*DY*inv01 + DY*DY*inv11)
    gauss = alpha_i * np.exp(np.clip(power, -20, 0))   # (h, w)

    # Back-to-front alpha composite
    g3 = gauss[:, :, np.newaxis]
    cr, cg, cb = color[i, 0], color[i, 1], color[i, 2]
    canvas[y0:y1, x0:x1, 0] = cr * gauss + canvas[y0:y1, x0:x1, 0] * (1 - gauss)
    canvas[y0:y1, x0:x1, 1] = cg * gauss + canvas[y0:y1, x0:x1, 1] * (1 - gauss)
    canvas[y0:y1, x0:x1, 2] = cb * gauss + canvas[y0:y1, x0:x1, 2] * (1 - gauss)

    if step % 5000 == 0:
        print(f"  {step}/{N} ...", flush=True)

# Convert to uint8 and save
img_np = np.clip(canvas[:, :, :3] * 255, 0, 255).astype(np.uint8)
img = Image.fromarray(img_np, "RGB")

out_path = os.path.join(OUT_DIR, "leaf_render.png")
img.save(out_path)
print(f"\nSaved → {out_path}")

# Also save a 2x-upscaled version for easier viewing
img_big = img.resize((IMG_W*2, IMG_H*2), Image.NEAREST)
out_big = os.path.join(OUT_DIR, "leaf_render_2x.png")
img_big.save(out_big)
print(f"Saved → {out_big}")
