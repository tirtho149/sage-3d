"""Frozen DINOv2 ViT-S/14 CLS features (cached) for any list of image paths.

Same preprocessing as SAGE-Extend's downstream test: Resize 256 (bicubic) + CenterCrop 224 + ImageNet norm;
L2-normalised. Cache: data/features/dinov2_vits14.npz (path -> feature), extended incrementally.
usage: CUDA_VISIBLE_DEVICES=0 ~/Desktop/Research/.venv/bin/python -m src.training.features <csv> [<csv> ...]
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HUB = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
CACHE = f"{ROOT}/data/features/dinov2_vits14.npz"
TF = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def load_cache():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=False)
        return dict(zip(z["paths"].tolist(), z["feats"]))
    return {}


def extract(paths, bs=128):
    cache = load_cache(); todo = [p for p in dict.fromkeys(paths) if p not in cache]
    if todo:
        model = torch.hub.load(str(HUB), "dinov2_vits14", source="local", pretrained=True).eval().cuda()
        with torch.no_grad():
            for i in range(0, len(todo), bs):
                x = torch.stack([TF(Image.open(os.path.join(ROOT, p)).convert("RGB")) for p in todo[i:i + bs]]).cuda()
                f = torch.nn.functional.normalize(model(x).float(), dim=1).cpu().numpy().astype(np.float16)
                cache.update(zip(todo[i:i + bs], f))
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.savez(CACHE, paths=np.array(list(cache)), feats=np.stack(list(cache.values())))
    return np.stack([cache[p] for p in paths]).astype(np.float32)


if __name__ == "__main__":
    paths = pd.concat([pd.read_csv(c) for c in sys.argv[1:]]).path.tolist()
    F = extract(paths); print("features", F.shape)
