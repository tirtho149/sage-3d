"""
Apply mango anthracnose disease symptoms to leaf_splat.ply.
Lesions: irregular dark-brown-to-black sunken spots, scattered naturally.
Healthy tissue: preserved original green.
"""
import numpy as np
import os
from PIL import Image

PLY_IN  = "/work/mech-ai-scratch/UshB/sage-extend/leaf_output/leaf_splat.ply"
OUT_DIR = "/work/mech-ai-scratch/UshB/sage-extend/Modified"
PLY_OUT = os.path.join(OUT_DIR, "leaf_anthracnose.ply")
PNG_OUT = os.path.join(OUT_DIR, "leaf_anthracnose.png")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load PLY ───────────────────────────────────────────────────────────────
with open(PLY_IN, "rb") as f:
    header_lines = []
    while True:
        line = f.readline()
        header_lines.append(line)
        if line.strip() == b"end_header":
            break
    header_bytes = b"".join(header_lines)
    raw = f.read()

PROPS = ["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity",
         "scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
N = 37120
arr = np.frombuffer(raw, dtype=np.dtype([(p, np.float32) for p in PROPS]), count=N).copy()

C0 = 0.28209479177387814

# ── lesion definitions ─────────────────────────────────────────────────────
# (cx, cy, major_r, minor_r, angle_deg, darkness_0to1)
# darkness: 1.0 = nearly black core, 0.6 = dark-brown only
LESIONS = [
    ( 0.01,  0.40, 0.030, 0.020,  15,  0.85),   # near tip
    (-0.06,  0.33, 0.024, 0.016, -18,  0.75),   # upper-left
    ( 0.07,  0.27, 0.021, 0.014,  42,  0.90),   # upper-right small
    ( 0.02,  0.17, 0.044, 0.029,   6,  0.80),   # middle-upper large
    (-0.08,  0.09, 0.027, 0.018, -12,  0.72),   # middle-left
    ( 0.07,  0.04, 0.017, 0.012,  32,  0.88),   # middle-right tiny
    (-0.02,  0.00, 0.040, 0.026,  22,  0.76),   # centre
    ( 0.09, -0.09, 0.019, 0.013, -38,  0.83),   # right-mid
    (-0.07, -0.16, 0.031, 0.020,   8,  0.70),   # lower-mid-left
    ( 0.03, -0.21, 0.046, 0.030, -14,  0.78),   # lower-mid large
    ( 0.08, -0.29, 0.021, 0.015,  28,  0.87),   # lower-right
    (-0.04, -0.36, 0.027, 0.017,  -4,  0.73),   # lower-left
    (-0.02,  0.45, 0.017, 0.011,  -8,  0.89),   # tip area
    ( 0.05,  0.12, 0.014, 0.010,  48,  0.91),   # satellite near large
    (-0.07,  0.21, 0.015, 0.010, -28,  0.82),   # upper-left satellite
    ( 0.04, -0.11, 0.019, 0.013,  17,  0.77),   # small mid
]

# Anthracnose brown-black color ramp (RGB 0-1)
# darkness=1 → nearly black; darkness=0 → dark brown
def lesion_color(darkness, norm_dist, rng_noise):
    """Return RGB for a point inside a lesion."""
    # Core is darker, edge is browner
    depth_factor = 1.0 - 0.5 * norm_dist          # 1 at centre, 0.5 at edge

    r_core = 0.07;  r_edge = 0.22
    g_core = 0.03;  g_edge = 0.10
    b_core = 0.01;  b_edge = 0.04

    r = (r_core + (r_edge - r_core) * (1 - depth_factor)) * (0.4 + 0.6 * (1 - darkness))
    g = (g_core + (g_edge - g_core) * (1 - depth_factor)) * (0.4 + 0.6 * (1 - darkness))
    b = (b_core + (b_edge - b_core) * (1 - depth_factor)) * (0.4 + 0.6 * (1 - darkness))

    # subtle per-Gaussian noise so lesion isn't flat
    r = np.clip(r + rng_noise * 0.03, 0.01, 0.30)
    g = np.clip(g + rng_noise * 0.02, 0.00, 0.15)
    b = np.clip(b + rng_noise * 0.01, 0.00, 0.06)
    return r, g, b

def irregular_ellipse_dist(dx, dy, angle_rad, major_r, minor_r, seed):
    """Normalised distance from ellipse boundary with organic edge noise."""
    rng = np.random.default_rng(seed)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    u =  cos_a * dx + sin_a * dy
    v = -sin_a * dx + cos_a * dy
    norm_dist = np.sqrt((u / major_r) ** 2 + (v / minor_r) ** 2 + 1e-12)
    theta = np.arctan2(v / minor_r, u / major_r)
    # Fourier harmonics for organic boundary
    noise_scale = 0.0
    for k, amp, phase in [
        (3, 0.13, rng.uniform(0, 2*np.pi)),
        (5, 0.07, rng.uniform(0, 2*np.pi)),
        (7, 0.04, rng.uniform(0, 2*np.pi)),
        (11,0.02, rng.uniform(0, 2*np.pi)),
    ]:
        noise_scale += amp * np.sin(k * theta + phase)
    threshold = 1.0 + noise_scale
    return norm_dist, threshold   # inside if norm_dist < threshold

# ── apply lesions ──────────────────────────────────────────────────────────
rng_global = np.random.default_rng(42)
px = arr["x"]; py = arr["y"]

lesion_mask      = np.zeros(N, dtype=bool)
lesion_norm_dist = np.ones(N, dtype=np.float32)
lesion_darkness  = np.zeros(N, dtype=np.float32)
lesion_noise_val = rng_global.uniform(-1, 1, N).astype(np.float32)

for li, (cx, cy, maj_r, min_r, angle_deg, darkness) in enumerate(LESIONS):
    angle_rad = np.deg2rad(angle_deg)
    dx = px - cx
    dy = py - cy
    nd, thresh = irregular_ellipse_dist(dx, dy, angle_rad, maj_r, min_r, seed=li*17+3)
    inside = nd < thresh
    # keep track of the lesion with minimum normalised distance per Gaussian
    update = inside & (~lesion_mask | (nd / thresh < lesion_norm_dist))
    lesion_mask[inside] = True
    lesion_norm_dist[update] = (nd / thresh)[update]
    lesion_darkness[update] = darkness

print(f"Gaussians in lesions: {lesion_mask.sum()} / {N}  ({100*lesion_mask.sum()/N:.1f}%)")

# Apply color changes
indices = np.where(lesion_mask)[0]
for idx in indices:
    nd   = float(lesion_norm_dist[idx])
    dark = float(lesion_darkness[idx])
    noise= float(lesion_noise_val[idx])
    r, g, b = lesion_color(dark, nd, noise)
    arr["f_dc_0"][idx] = np.float32((r - 0.5) / C0)
    arr["f_dc_1"][idx] = np.float32((g - 0.5) / C0)
    arr["f_dc_2"][idx] = np.float32((b - 0.5) / C0)
    # Sunken: shift slightly in -Z and compress Z scale
    arr["z"][idx]       = arr["z"][idx] - np.float32(0.004 * dark)
    arr["scale_2"][idx] = arr["scale_2"][idx] + np.float32(np.log(0.75))

print("Lesion colors applied.")

# ── save PLY ───────────────────────────────────────────────────────────────
with open(PLY_OUT, "wb") as f:
    f.write(header_bytes)
    f.write(arr.tobytes())
print(f"Saved PLY → {PLY_OUT}")

# ── render ─────────────────────────────────────────────────────────────────
print("\nRendering PNG (orthographic Gaussian splatting) ...")

pos   = np.stack([arr["x"], arr["y"], arr["z"]], axis=1)
color = np.clip(np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=1)*C0+0.5, 0, 1)
opa   = (1.0/(1.0+np.exp(-arr["opacity"].astype(np.float64)))).astype(np.float32)

scales_exp = np.exp(np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], axis=1))
quats = np.stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]], axis=1)
quats = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-8)

# Vectorised 2D covariance (orthographic, project onto XY plane)
w, qx, qy, qz = quats[:,0], quats[:,1], quats[:,2], quats[:,3]
R3 = np.stack([
    1-2*(qy*qy+qz*qz), 2*(qx*qy-w*qz),   2*(qx*qz+w*qy),
    2*(qx*qy+w*qz),   1-2*(qx*qx+qz*qz),  2*(qy*qz-w*qx),
    2*(qx*qz-w*qy),   2*(qy*qz+w*qx),   1-2*(qx*qx+qy*qy)
], axis=1).reshape(N, 3, 3)
M   = R3 * scales_exp[:, np.newaxis, :]
c00 = (M[:,0,:] ** 2).sum(1)
c11 = (M[:,1,:] ** 2).sum(1)
c01 = (M[:,0,:] * M[:,1,:]).sum(1)

X_MIN = pos[:,0].min()-0.012; X_MAX = pos[:,0].max()+0.012
Y_MIN = pos[:,1].min()-0.012; Y_MAX = pos[:,1].max()+0.012
IMG_W = 700
IMG_H = int(IMG_W * (Y_MAX - Y_MIN) / (X_MAX - X_MIN))
PX = IMG_W / (X_MAX - X_MIN)
PY = IMG_H / (Y_MAX - Y_MIN)

sc00 = c00 * PX*PX; sc11 = c11 * PY*PY; sc01 = c01 * PX*PY
cx_s = ((pos[:,0]-X_MIN)*PX).astype(np.float32)
cy_s = ((Y_MAX-pos[:,1])*PY).astype(np.float32)

order = np.argsort(pos[:,2])   # back-to-front

canvas = np.ones((IMG_H, IMG_W, 3), dtype=np.float32)  # white bg

SIGMA_CLIP = 3.0
MIN_ALPHA  = 0.01

for step, i in enumerate(order):
    ai = float(opa[i])
    if ai < MIN_ALPHA:
        continue
    det  = sc00[i]*sc11[i] - sc01[i]**2 + 1e-8
    lam1 = 0.5*((sc00[i]+sc11[i]) + np.sqrt((sc00[i]-sc11[i])**2 + 4*sc01[i]**2 + 1e-8))
    rad  = SIGMA_CLIP * np.sqrt(abs(lam1))
    x0 = max(0, int(cx_s[i]-rad));     x1 = min(IMG_W, int(cx_s[i]+rad)+1)
    y0 = max(0, int(cy_s[i]-rad));     y1 = min(IMG_H, int(cy_s[i]+rad)+1)
    if x0>=x1 or y0>=y1: continue
    dx_ = np.arange(x0,x1,dtype=np.float32)-cx_s[i]
    dy_ = np.arange(y0,y1,dtype=np.float32)-cy_s[i]
    DX, DY = np.meshgrid(dx_, dy_)
    inv00 =  sc11[i]/det; inv11 = sc00[i]/det; inv01 = -sc01[i]/det
    power = -0.5*(DX*DX*inv00 + 2*DX*DY*inv01 + DY*DY*inv11)
    gauss = ai * np.exp(np.clip(power, -20, 0))
    cr,cg,cb = color[i,0], color[i,1], color[i,2]
    canvas[y0:y1,x0:x1,0] = cr*gauss + canvas[y0:y1,x0:x1,0]*(1-gauss)
    canvas[y0:y1,x0:x1,1] = cg*gauss + canvas[y0:y1,x0:x1,1]*(1-gauss)
    canvas[y0:y1,x0:x1,2] = cb*gauss + canvas[y0:y1,x0:x1,2]*(1-gauss)
    if step % 8000 == 0:
        print(f"  {step}/{N} ...", flush=True)

img = Image.fromarray(np.clip(canvas*255,0,255).astype(np.uint8), "RGB")
img.save(PNG_OUT)
print(f"\nSaved PNG → {PNG_OUT}")
