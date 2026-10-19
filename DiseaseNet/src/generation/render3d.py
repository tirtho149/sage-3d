"""Stage 0 step 5: 3D arms of the synthetic dataset (sam3d env, one GPU).

Per parent asset:
  1. input-camera fit with SAGE-Extend's alignment (camera_alignment/align.py main: intrinsics from the
     SAM 3D pointmap, A2 closed-form + A3 render-and-compare pose correction); pilot assets reuse theirs.
  2. arms
       U3D   unedited asset
       KB3D  KB geometry operators where implemented (kb_specs.json; deform.run with SAGE-Extend defaults),
             then the KB 2D edit (edits2d/<asset>/kb_s0.png) lifted onto the Gaussians visible at the input
             camera (depth-tested); hidden Gaussians keep their colour
  3. 8 photo-like views: the corrected input camera and object rotations about its centroid
     (yaw/pitch in degrees) VIEWS, at the photo's own resolution and framing
  4. 4 rendering conditions
       R0 white background
       R1 own background plate (the root photo with the organ inpainted, cv2 Telea)
       R2 background plate of another Stage-0 parent photo, uniform over classes (class-independent)
       R5 R2 + photometric/camera variation (brightness, contrast, white balance, blur, noise, JPEG)
Outputs JPEGs under data/synthetic/stage0/images/<arm>/<asset>/v<k>_<cond>.jpg and rows appended to
data/synthetic/stage0/manifest_3d.csv (path, arm, class_id, asset_id, root_sha1, view, cond, input_iou).
usage: CUDA_VISIBLE_DEVICES=1 LIDRA_SKIP_INIT=true ~/micromamba/envs/sam3d/bin/python src/generation/render3d.py
"""
import csv
import glob
import json
import os
import sys
import zlib

os.environ.setdefault("LIDRA_SKIP_INIT", "true")
os.environ.setdefault("CONDA_PREFIX", os.path.expanduser("~/micromamba/envs/sam3d"))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.expanduser("~/Desktop/Research/external/SAGE-Extend")
sys.path.insert(0, f"{EXT}/src"); sys.path.insert(0, f"{EXT}/src/camera_alignment")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from plyfile import PlyData, PlyElement  # noqa: E402

ST = f"{ROOT}/data/synthetic/stage0"
VIEWS = [(0, 0), (15, 0), (-15, 0), (0, 15), (0, -15), (30, 0), (-30, 0), (0, 25)]
CONDS = ["R0", "R1", "R2", "R5"]
OUT_MAX = 384                                          # long side of saved images (classifier crops 224)
EXT_PILOT_ALIGN = f"{EXT}/results/alignment"


def seed_of(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def load_inputs(a):
    img = np.asarray(Image.open(a["image"]).convert("RGB")); m = np.asarray(Image.open(a["mask"]).convert("L")) > 127
    return img, m


def plate(img, mask):
    k = max(3, int(0.02 * max(mask.shape)) | 1)
    mm = cv2.dilate(mask.astype(np.uint8) * 255, np.ones((k, k), np.uint8))
    return cv2.inpaint(img, mm, 7, cv2.INPAINT_TELEA)


def alignment(a, in_dir):
    if a["origin"] == "sage-extend-pilot":
        return json.load(open(f"{EXT_PILOT_ALIGN}/{a['asset_id'].replace('pilot__', '')}/alignment.json"))
    out = f"{ST}/alignment/{a['asset_id']}"
    if not os.path.exists(f"{out}/alignment.json"):
        import align
        align.main(a["run_dir"], in_dir, out)
    return json.load(open(f"{out}/alignment.json"))


def posed(a, ply, A):
    import align
    G, _ = align.posed_gaussians(a["run_dir"], ply)
    m = A["metrics"]; sel = "A3" if m["A3_render_compare"]["iou"] > m["A2_closed_form"]["iou"] else "A2"
    pc = A["pose_correction"][sel]; c = torch.tensor(A["pose_correction"]["centroid_obj_cam"], device=align.DEV)
    G = align.transform(G, torch.tensor(pc["rot_axis_angle"], device=align.DEV), torch.tensor(pc["t"], device=align.DEV),
                        torch.tensor(pc["log_scale"], device=align.DEV), c)
    return G, sel


def rotate(G, yaw, pitch):
    import align
    c = G["means"].mean(0); z = torch.zeros(3, device=align.DEV)
    Gr = align.transform(G, torch.tensor([0.0, np.radians(yaw), 0.0], device=align.DEV, dtype=torch.float32), z, torch.zeros((), device=align.DEV), c)
    return align.transform(Gr, torch.tensor([np.radians(pitch), 0.0, 0.0], device=align.DEV, dtype=torch.float32), z, torch.zeros((), device=align.DEV), c)


def render(G, K, W, H):
    from gsplat import rasterization
    with torch.no_grad():
        out, al, _ = rasterization(G["means"], G["quats"], G["scales"], G["op"], G["rgb"], torch.eye(4, device="cuda")[None],
                                   K[None], W, H, render_mode="RGB+ED")
    return out[0, ..., :3].clamp(0, 1).cpu().numpy(), al[0, ..., 0].cpu().numpy(), out[0, ..., 3].cpu().numpy()


def lift_colors(G, K, W, H, edit_img):
    """Colour of Gaussians visible at the input camera <- edited photo pixel; hidden ones unchanged."""
    _, alpha, depth = render(G, K, W, H)
    X = G["means"].cpu().numpy(); k = K.cpu().numpy(); z = X[:, 2]
    u = np.round(k[0, 0] * X[:, 0] / z + k[0, 2]).astype(int); v = np.round(k[1, 1] * X[:, 1] / z + k[1, 2]).astype(int)
    ok = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    L = float(np.linalg.norm(X.max(0) - X.min(0)))
    vis = np.zeros(len(X), bool)
    d = np.where(alpha > 1e-3, depth / np.maximum(alpha, 1e-3), np.inf)
    vis[ok] = (z[ok] <= d[v[ok], u[ok]] + 0.01 * L) & (alpha[v[ok], u[ok]] > 0.5)
    ed = np.asarray(Image.fromarray(edit_img).resize((W, H)), np.float32) / 255
    rgb = G["rgb"].clone(); rgb[torch.from_numpy(vis).cuda()] = torch.from_numpy(ed[v[vis], u[vis]]).cuda()
    return rgb, float(vis.mean())


def geometry_edit(a, spec):
    ops = spec["geometry"]["operators"]
    if not ops:
        return None, []
    out = f"{ST}/edited/{a['asset_id']}/geo.ply"
    if not os.path.exists(out):
        from geometry_editor.deform import run as deform
        D = json.load(open(f"{EXT}/configs/geometry/operator_defaults.json"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        deform(a["splat"], dict(seed=0, ops=[dict(D[g["op"]], op=g["op"], kb_term=g["term"], kb_phrase=g["phrase"]) for g in ops], iters=300), out)
    return out, [g["op"] for g in ops]


def photometric(im, rng):
    x = im.astype(np.float32)
    x = x * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)                                   # brightness/contrast
    x = x * rng.uniform(0.9, 1.1, size=3)                                                     # white balance
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.5))
    x = np.clip(x + rng.normal(0, rng.uniform(0, 6), x.shape), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", x[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(40, 90))])
    return cv2.imdecode(enc, 1)[..., ::-1]


def compose(rgb, alpha, bg):
    a = alpha[..., None]
    b = np.asarray(Image.fromarray(bg).resize((rgb.shape[1], rgb.shape[0])), np.float32) / 255
    return ((rgb * a + b * (1 - a)) * 255).clip(0, 255).astype(np.uint8)


def crop_box(alpha, margin=0.15):
    """Square box around the rendered object (+margin), clipped to the image: photo-like organ scale."""
    ys, xs = np.nonzero(alpha > 0.5)
    H, W = alpha.shape
    if len(xs) == 0:
        return 0, 0, W, H
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * margin)
    side = min(side, max(W, H))
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    return x0, y0, int(min(W, x0 + side)), int(min(H, y0 + side))


def save(im, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = im.shape[:2]; s = OUT_MAX / max(h, w)
    Image.fromarray(im).resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR).save(path, quality=92)


def main():
    sys.path.insert(0, ROOT)
    from src.generation.registry import assets
    A_all = assets(); A = A_all[A_all.splat.notna()]
    K = json.load(open(f"{ST}/kb_specs.json"))
    man_path = f"{ST}/manifest_3d.csv"; done = set()
    if os.path.exists(man_path):
        with open(man_path) as f:
            done = {(r["arm"], r["asset_id"]) for r in csv.DictReader(f)}
    plates = {}
    for _, a in A_all.iterrows():                      # plates from ALL parents, independent of render timing
        img, m = load_inputs(a); plates[a.asset_id] = plate(img, m)
    ids = sorted(plates); cls_of = dict(zip(A_all.asset_id, A_all.class_id))
    with open(man_path, "a", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["path", "arm", "class_id", "asset_id", "root_sha1", "view", "cond", "bg_asset", "input_iou", "geo_ops", "lift_frac"])
        if not done:
            w.writeheader()
        for _, a in A.iterrows():
            a = a.to_dict()
            if all((arm, a["asset_id"]) in done for arm in ("U3D", "KB3D")):
                continue
            img, m = load_inputs(a); H, W = m.shape
            in_dir = os.path.dirname(a["image"]); Al = alignment(a, in_dir)
            Kt = torch.tensor(Al["K"], dtype=torch.float32, device="cuda")
            geo_ply, ops = geometry_edit(a, K[a["class_id"]])
            ed_path = f"{ST}/edits2d/{a['asset_id']}/kb_s0.png"
            for arm in ("U3D", "KB3D"):
                if (arm, a["asset_id"]) in done:
                    continue
                if arm == "KB3D" and not os.path.exists(ed_path):
                    print("skip KB3D (no 2D KB edit yet)", a["asset_id"], flush=True); continue
                ply = a["splat"] if arm == "U3D" or not geo_ply else geo_ply
                G, sel = posed(a, ply, Al); lf = ""
                if arm == "KB3D":
                    G["rgb"], lf = lift_colors(G, Kt, W, H, np.asarray(Image.open(ed_path).convert("RGB")))
                _, a0, _ = render(G, Kt, W, H); iou = float(((a0 > 0.5) & m).sum() / max(((a0 > 0.5) | m).sum(), 1))
                for vi, (yaw, pitch) in enumerate(VIEWS):
                    rgb, alpha, _ = render(rotate(G, yaw, pitch) if (yaw or pitch) else G, Kt, W, H)
                    x0, y0, x1, y1 = crop_box(alpha)
                    for cond in CONDS:
                        rng = np.random.default_rng(seed_of(arm, a["asset_id"], vi, cond))
                        bg_id = ""
                        if cond == "R0":
                            im = compose(rgb, alpha, np.full((8, 8, 3), 255, np.uint8))
                        elif cond == "R1":
                            bg_id = a["asset_id"]; im = compose(rgb, alpha, plates[bg_id])
                        else:
                            others = [i for i in ids if cls_of[i] != a["class_id"]]   # class-independent background
                            bg_id = others[int(rng.integers(len(others)))]; im = compose(rgb, alpha, plates[bg_id])
                            if cond == "R5":
                                im = photometric(im, rng)
                        p = f"{ST}/images/{arm}/{a['asset_id']}/v{vi}_{cond}.jpg"; save(im[y0:y1, x0:x1], p)
                        w.writerow(dict(path=os.path.relpath(p, ROOT), arm=arm, class_id=a["class_id"], asset_id=a["asset_id"],
                                        root_sha1=a["root_sha1"], view=vi, cond=cond, bg_asset=bg_id, input_iou=round(iou, 4),
                                        geo_ops="+".join(ops) if arm == "KB3D" else "", lift_frac=lf))
                fo.flush()
                print(arm, a["asset_id"], f"pose={sel} input_iou={iou:.2f} ops={ops if arm == 'KB3D' else []}", flush=True)


if __name__ == "__main__":
    main()
