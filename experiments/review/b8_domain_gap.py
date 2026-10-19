"""Review item B8: small matched pilot that isolates where the render-to-real gap comes from.

For each of the 15 pilot specimens, frozen DINOv2 ViT-B/14 CLS features (same model and
preprocessing as src/downstream/features.py) of:
  photo        the real input photo (sanity reference)
  in_photo     unedited asset rendered at the corrected INPUT camera, composited over the
               photo's own background  (pose + background matched: only the object rendering differs)
  in_white     the same render on white                         (+ background change)
  rig_white    unedited asset at 6 rig views, white             (+ viewpoint change)
  rig_phototex asset whose input-visible Gaussians take the REAL photo colours, 6 rig views, white
               (texture replaced by the photo's: does the gap shrink?)
Metric: cosine distance to the nearest REAL photo of the same pilot class (build pool of
the downstream test), and real-vs-condition AUC (5-fold logistic regression), as in A12.
Each step changes one factor, so the step-to-step change is that factor's share of the gap.
Descriptive pilot (15 specimens); nothing is tuned.

usage (sam3d env): CUDA_VISIBLE_DEVICES=1 LIDRA_SKIP_INIT=true ~/micromamba/envs/sam3d/bin/python experiments/review/b8_domain_gap.py
  -> results/review/b8_domain_gap/summary.{json,md}
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from torchvision import transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/src"); sys.path.insert(0, f"{ROOT}/experiments/demos")
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
from geometry_editor.anatomy import input_camera, rgb_to_dc  # noqa: E402
from camera_alignment.align import render_at_input  # noqa: E402
from rendering.rig_render import render_rig  # noqa: E402
from run_demos import rig_of  # noqa: E402

OUT = f"{ROOT}/results/review/b8_domain_gap"
DSM = Path(f"{ROOT}/results/downstream_minimal")
HUB = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
RIG_VIEWS = [0, 2, 4, 6, 16, 17]
CONDS = ["photo", "in_photo", "in_white", "rig_white", "rig_phototex"]


def phototex(sid, d):
    ply = f"{ROOT}/results/pilot/sam3d/{sid}/seed0/splat_ge.ply"; cam = input_camera(sid, ply)
    img = np.asarray(Image.open(f"{ROOT}/data/inputs/{sid}/image.png").convert("RGB").resize((cam["W"], cam["H"])), np.float64) / 255
    uvw = cam["Xc"] @ np.asarray(cam["K"]).T
    px = np.clip(np.round(uvw[:, 0] / uvw[:, 2]).astype(int), 0, cam["W"] - 1)
    py = np.clip(np.round(uvw[:, 1] / uvw[:, 2]).astype(int), 0, cam["H"] - 1)
    vis = cam["visible"]; v = PlyData.read(ply)["vertex"].data.copy(); dc = rgb_to_dc(img[py[vis], px[vis]])
    for k in range(3):
        v[f"f_dc_{k}"][vis] = dc[:, k].astype(v[f"f_dc_{k}"].dtype)
    out = f"{d}/phototex.ply"; PlyData([PlyElement.describe(v, "vertex")], text=False).write(out); return out


def images(sid):
    d = f"{OUT}/{sid}"; os.makedirs(d, exist_ok=True)
    ply = f"{ROOT}/results/pilot/sam3d/{sid}/seed0/splat_ge.ply"
    photo = np.asarray(Image.open(f"{ROOT}/data/inputs/{sid}/image.png").convert("RGB"), np.float64)
    rgb, a = render_at_input(ply, sid)
    H, W = a.shape[:2]; photo_r = np.asarray(Image.fromarray(photo.astype(np.uint8)).resize((W, H)), np.float64)
    a3 = np.clip(a, 0, 1)[..., None] if a.ndim == 2 else np.clip(a, 0, 1)
    obj = rgb.astype(np.float64)
    ims = {"photo": [photo.astype(np.uint8)],
           "in_photo": [(obj * a3 + photo_r * (1 - a3)).clip(0, 255).astype(np.uint8)],
           "in_white": [(obj * a3 + 255 * (1 - a3)).clip(0, 255).astype(np.uint8)],
           "rig_white": [r["rgb"] for r in render_rig(ply, rig_of(sid), views=RIG_VIEWS)],
           "rig_phototex": [r["rgb"] for r in render_rig(phototex(sid, d), rig_of(sid), views=RIG_VIEWS)]}
    for c, L in ims.items():
        for i, im in enumerate(L):
            Image.fromarray(im).save(f"{d}/{c}_{i}.png")
    return ims


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"
    model = torch.hub.load(str(HUB), "dinov2_vitb14", source="local", pretrained=True).eval().to(dev)
    tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def feat(arrs):
        with torch.no_grad():
            x = torch.stack([tf(Image.fromarray(a).convert("RGB")) for a in arrs]).to(dev)
            return torch.nn.functional.normalize(model(x).float(), dim=1).cpu().numpy()

    z = np.load(DSM / "features.npz"); fmap = {p: i for i, p in enumerate(z["paths"])}
    Fz = z["feats"].astype(np.float32); Fz /= np.linalg.norm(Fz, axis=1, keepdims=True) + 1e-8
    real = pd.read_csv(DSM / "image_manifest_real.csv"); real["fi"] = real.path.map(fmap); real = real.dropna(subset=["fi"])
    ref = real[real.is_pilot_class.astype(bool) & (real.split == "build")]
    sids = sorted(os.path.basename(p) for p in glob.glob(f"{ROOT}/results/pilot/*__*") if os.path.isdir(p) and "sam3d" not in p)
    rows, feats = [], {c: [] for c in CONDS}
    for sid in sids:
        crop = sid.split("__")[0]
        R = Fz[ref[ref.crop == crop].fi.astype(int).values]
        if len(R) == 0:
            continue
        ims = images(sid)
        for c in CONDS:
            f = feat(ims[c]); feats[c].append(f)
            d = 1 - (f @ R.T).max(1)
            rows.append(dict(specimen=sid, cond=c, nn_dist=float(np.median(d))))
        print(sid, {r["cond"]: round(r["nn_dist"], 3) for r in rows if r["specimen"] == sid}, flush=True)
    df = pd.DataFrame(rows)
    med = df.groupby("cond").nn_dist.median().reindex(CONDS)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    allref = Fz[ref.fi.astype(int).values]
    auc = {}
    for c in CONDS:
        S = np.concatenate(feats[c]); n = min(len(S), len(allref))
        Rr = allref[np.random.default_rng(0).choice(len(allref), n, replace=False)]
        X = np.concatenate([Rr, S]); y = np.r_[np.zeros(len(Rr)), np.ones(len(S))]
        k = min(5, int(min(len(Rr), len(S))))
        auc[c] = float(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=StratifiedKFold(k, shuffle=True, random_state=0),
                                       scoring="roc_auc").mean()) if k >= 2 else None
    out = dict(per_specimen=rows, median_nn_dist=med.to_dict(), auc=auc,
               steps={"render object at matched pose+background (photo -> in_photo)": float(med["in_photo"] - med["photo"]),
                      "background (in_photo -> in_white)": float(med["in_white"] - med["in_photo"]),
                      "viewpoint (in_white -> rig_white)": float(med["rig_white"] - med["in_white"]),
                      "texture = photo colours (rig_white -> rig_phototex)": float(med["rig_phototex"] - med["rig_white"])})
    json.dump(out, open(f"{OUT}/summary.json", "w"), indent=1)
    L = ["# B8: where the render-to-real gap comes from (matched pilot, 15 specimens)", "",
         "Median cosine distance to the nearest real build-pool photo of the same pilot class (DINOv2 ViT-B/14), and",
         "real-vs-condition AUC. Each row changes one factor relative to the row above.", "",
         "| condition | what changes | median NN distance | AUC vs real |", "|---|---|---|---|"]
    what = {"photo": "real photo (reference)", "in_photo": "object rendered at the photo's pose, photo background",
            "in_white": "+ white background", "rig_white": "+ new viewpoints", "rig_phototex": "texture replaced by photo colours"}
    for c in CONDS:
        L.append(f"| {c} | {what[c]} | {med[c]:.3f} | {auc[c]:.2f} |" if auc[c] is not None else f"| {c} | {what[c]} | {med[c]:.3f} | - |")
    L += ["", "Step changes in median distance:"] + [f"- {k}: {v:+.3f}" for k, v in out["steps"].items()]
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
