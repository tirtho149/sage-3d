"""Edit leaf_splat.ply: turn leaf yellow and punch 5 opacity holes."""
import numpy as np
import os

PLY_IN  = "/work/mech-ai-scratch/UshB/sage-extend/leaf_output/leaf_splat.ply"
PLY_OUT_DIR = "/work/mech-ai-scratch/UshB/sage-extend/Modified"
PLY_OUT = os.path.join(PLY_OUT_DIR, "leaf_yellow_holes.ply")
os.makedirs(PLY_OUT_DIR, exist_ok=True)

# ---- parse header -------------------------------------------------------
with open(PLY_IN, "rb") as f:
    header_lines = []
    while True:
        line = f.readline()
        header_lines.append(line)
        if line.strip() == b"end_header":
            break
    header_bytes = b"".join(header_lines)
    raw_data = f.read()

PROPS = [
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]
N_VERTS = 37120
dt = np.dtype([(p, np.float32) for p in PROPS])
arr = np.frombuffer(raw_data, dtype=dt, count=N_VERTS).copy()

# ---- make leaf yellow ---------------------------------------------------
C0 = 0.28209479177387814

rng = np.random.default_rng(0)
# Slight per-Gaussian variation so it looks natural, not flat plastic
noise = rng.uniform(-0.04, 0.04, (N_VERTS, 3)).astype(np.float32)

target_r = np.clip(0.85 + noise[:, 0], 0.0, 1.0)
target_g = np.clip(0.72 + noise[:, 1], 0.0, 1.0)
target_b = np.clip(0.08 + noise[:, 2], 0.0, 1.0)

arr["f_dc_0"] = ((target_r - 0.5) / C0).astype(np.float32)
arr["f_dc_1"] = ((target_g - 0.5) / C0).astype(np.float32)
arr["f_dc_2"] = ((target_b - 0.5) / C0).astype(np.float32)

print(f"Yellow applied — R mean: {target_r.mean():.3f}, G mean: {target_g.mean():.3f}, B mean: {target_b.mean():.3f}")

# ---- punch 5 holes (zero opacity in circular XY regions) ----------------
# Leaf spans X: -0.14..0.14, Y: -0.50..0.50
# Place 5 holes in a natural scattered pattern, avoiding edges
hole_centers = np.array([
    [ 0.00,  0.32],   # top-center
    [-0.07,  0.12],   # upper-left
    [ 0.07,  0.12],   # upper-right
    [-0.06, -0.15],   # lower-left
    [ 0.06, -0.15],   # lower-right
], dtype=np.float32)

HOLE_RADIUS = 0.038   # ~14% of leaf half-width — visible but not giant

x = arr["x"]
y = arr["y"]
total_masked = 0
for i, (hx, hy) in enumerate(hole_centers):
    dist2 = (x - hx) ** 2 + (y - hy) ** 2
    inside = dist2 < HOLE_RADIUS ** 2
    arr["opacity"][inside] = np.float32(-10.0)   # sigmoid(-10) ≈ 0.000045 → invisible
    count = inside.sum()
    total_masked += count
    print(f"Hole {i+1}: center ({hx:.3f}, {hy:.3f}), radius {HOLE_RADIUS:.3f} — {count} Gaussians zeroed")

print(f"Total Gaussians in holes: {total_masked} / {N_VERTS} ({100*total_masked/N_VERTS:.1f}%)")

# ---- write output PLY ---------------------------------------------------
with open(PLY_OUT, "wb") as f:
    f.write(header_bytes)
    f.write(arr.tobytes())

size_kb = os.path.getsize(PLY_OUT) / 1024
print(f"\nSaved → {PLY_OUT}  ({size_kb:.0f} KB)")
