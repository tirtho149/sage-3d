"""Review item B7: a KB-constrained 2D geometric warp baseline for directional curl.

The strongest 2D shape baseline for "curl upwards" is the BEST CASE a 2D method could reach:
the exact projection of the 3D curl displacement (curl.curl_target, same theta, same r0)
into the photo, applied as a backward image warp. It changes the silhouette (margins
foreshorten), it receives the same KB constraint, and it needs no diffusion model.

Question: can a 2D representation express the cited DIRECTION? Curl up and curl down move
the margins toward and away from the camera; in the image plane both foreshorten almost the
same way. Measured per leaf:
  * dE_updown_2D: mean colour difference between the up-warp and the down-warp (input view);
  * dE_updown_3D_input / _rig: the same for the 3D up/down assets at the input view and averaged
    over the 18 rig views (renders);
  * silhouette IoU of each 2D warp vs the real mask (does the 2D warp also contradict the photo?).
Leaves: Mango 0044/0049/0094 (3D up/down assets from Test 1, seed 0).

usage (sam3d env): CUDA_VISIBLE_DEVICES=1 LIDRA_SKIP_INIT=true ~/micromamba/envs/sam3d/bin/python experiments/review/b7_warp2d.py
  -> results/review/b7_warp2d/<sid>/{up,down}.png, summary.{json,md}
"""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from skimage.color import rgb2lab

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/experiments/demos")
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
from geometry_editor.anatomy import leaf_frame  # noqa: E402
from geometry_editor.curl import curl_target, parse  # noqa: E402
from camera_alignment.align import render_at_input  # noqa: E402
from rendering.rig_render import render_rig  # noqa: E402
from run_demos import rig_of  # noqa: E402

OUT = f"{ROOT}/results/review/b7_warp2d"
LEAVES = ["Mango__Anthracnose__0044", "Mango__Anthracnose__0049", "Mango__Anthracnose__0094"]
PHRASE = "Leaves curl upwards and fall off."


def warp(sid, F, direction, P):
    cam = F["cam"]; s, R, t, K, W, H = cam["s"], cam["R"], cam["t"], np.asarray(cam["K"]), cam["W"], cam["H"]
    vis = cam["visible"]
    X = F["X"]; Y = X + curl_target(F, direction, P["theta_deg"], P["r0"])
    proj = lambda Z: (lambda c: c[:, :2] / c[:, 2:3])((s * Z @ R.T + t) @ K.T)
    p0, p1 = proj(X[vis]), proj(Y[vis])                      # where each visible surface point is / goes in the image
    img = np.asarray(Image.open(f"{ROOT}/data/inputs/{sid}/image.png").convert("RGB").resize((W, H)), np.float64)
    gy, gx = np.mgrid[0:H, 0:W]
    # backward map: for each output pixel (at a moved position p1) sample the photo at p0; identity elsewhere
    src = griddata(p1, p0, (gx, gy), method="linear")
    inside = ~np.isnan(src[..., 0])
    sx = np.where(inside, src[..., 0], gx); sy = np.where(inside, src[..., 1], gy)
    out = np.stack([map_coordinates(img[..., c], [sy, sx], order=1, mode="nearest") for c in range(3)], -1)
    return out.clip(0, 255).astype(np.uint8), inside


def mask_iou(inside, sid, W, H):
    m = np.asarray(Image.open(f"{ROOT}/data/inputs/{sid}/0.png").convert("L").resize((W, H))) > 127
    return float((inside & m).sum() / max((inside | m).sum(), 1))


def dE(a, b, m=None):
    A = rgb2lab(a / 255.0); B = rgb2lab(b / 255.0); d = np.linalg.norm(A - B, axis=-1)
    return float(d[m].mean()) if m is not None and m.any() else float(d.mean())


def main():
    os.makedirs(OUT, exist_ok=True)
    P = parse(PHRASE); res = {}
    for sid in LEAVES:
        F = leaf_frame(sid); d = f"{OUT}/{sid}"; os.makedirs(d, exist_ok=True)
        up, in_up = warp(sid, F, P["direction"], P); dn, in_dn = warp(sid, F, -P["direction"], P)
        Image.fromarray(up).save(f"{d}/up.png"); Image.fromarray(dn).save(f"{d}/down.png")
        W, H = F["cam"]["W"], F["cam"]["H"]
        a3u = f"{ROOT}/results/demos/mango_curl/{sid}/CU_kb_s0/asset.ply"; a3d = f"{ROOT}/results/demos/mango_curl/{sid}/CD_opposite_s0/asset.ply"
        ru, au = render_at_input(a3u, sid); rd, ad = render_at_input(a3d, sid)
        both_in = (au > 0.5) | (ad > 0.5)
        rig = rig_of(sid)
        Ru = render_rig(a3u, rig); Rd = render_rig(a3d, rig)
        rig_de = float(np.mean([dE(x["rgb"].astype(float), y["rgb"].astype(float), (x["alpha"] > 0.5) | (y["alpha"] > 0.5))
                                for x, y in zip(Ru, Rd)]))
        res[sid] = dict(dE_updown_2D=dE(up.astype(float), dn.astype(float), in_up | in_dn),
                        dE_updown_3D_input=dE(ru.astype(float), rd.astype(float), both_in),
                        dE_updown_3D_rig=rig_de,
                        iou_2D_up=mask_iou(in_up, sid, W, H), iou_2D_down=mask_iou(in_dn, sid, W, H))
        print(sid, {k: round(v, 3) for k, v in res[sid].items()}, flush=True)
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
    L = ["# B7: best-case 2D warp (projection of the 3D curl) vs 3D, for the cited curl direction", "",
         "dE up-vs-down = mean colour difference between the 'curl up' and 'curl down' results (object pixels).",
         "A representation that cannot tell the two directions apart scores ~0.", "",
         "| leaf | 2D warp, input view | 3D, input view | 3D, mean over 18 rig views | 2D warp IoU vs real mask (up / down) |",
         "|---|---|---|---|---|"]
    for sid, r in res.items():
        L.append(f"| {sid} | {r['dE_updown_2D']:.1f} | {r['dE_updown_3D_input']:.1f} | {r['dE_updown_3D_rig']:.1f} | "
                 f"{r['iou_2D_up']:.2f} / {r['iou_2D_down']:.2f} |")
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
