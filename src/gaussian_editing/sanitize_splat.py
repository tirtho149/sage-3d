"""Make a SAM 3D splat safe for GaussianEditor without changing what it renders.

Bug (CONFIRMED, reports/00_full_failure_audit.md F-GE-2): SAM 3D exports opacity logits of +inf for Gaussians with
opacity == 1.0. GaussianEditor's opacity anchor loss is then NaN, and with precision 16-mixed the GradScaler skips
EVERY optimizer step -> the "edit" returns the input unchanged (Peach 0006). Fix: replace non-finite opacity logits
by the largest finite logit (15.9424 = logit(1 - 2^-23), the value SAM 3D already uses for saturated Gaussians;
rendered opacity identical to fp32 precision). Any other non-finite field is an error (not silently repaired).

usage: sanitize_splat.py <splat.ply> <out.ply>   (writes <out.ply>.json with counts)
"""
import sys, json
import numpy as np
from plyfile import PlyData, PlyElement

OP_MAX = float(np.log((1 - 2.0 ** -23) / 2.0 ** -23))   # 15.9424


def sanitize(src, dst):
    ply = PlyData.read(src); v = ply["vertex"].data.copy()
    bad = {k: int((~np.isfinite(v[k])).sum()) for k in v.dtype.names if (~np.isfinite(v[k])).sum()}
    other = {k: n for k, n in bad.items() if k != "opacity"}
    if other: raise ValueError(f"non-finite values outside opacity: {other}")
    op = v["opacity"]; pos_inf, neg_inf, nan = int(np.isposinf(op).sum()), int(np.isneginf(op).sum()), int(np.isnan(op).sum())
    if nan: raise ValueError(f"{nan} NaN opacities")
    v["opacity"] = np.clip(op, -OP_MAX, OP_MAX)
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(dst)
    rep = dict(src=src, n=len(v), opacity_pos_inf=pos_inf, opacity_neg_inf=neg_inf, clamp=OP_MAX)
    json.dump(rep, open(dst + ".json", "w"), indent=1); return rep


if __name__ == "__main__":
    print(json.dumps(sanitize(sys.argv[1], sys.argv[2])))
