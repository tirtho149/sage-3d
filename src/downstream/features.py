"""Stage 2: frozen DINOv2 ViT-B/14 CLS features for every image in the manifests (GPU 0 only).

    CUDA_VISIBLE_DEVICES=0 ~/micromamba/envs/sam3d/bin/python -m src.downstream.features
Uses the local torch.hub checkout + cached checkpoint (no download). Writes
results/downstream_minimal/features.npz (float16, L2-normalised) keyed by image path.
"""
import os
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torchvision import transforms as T
from .common import OUT

HUB = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"


def main():
    real = pd.read_csv(OUT / "image_manifest_real.csv")
    syn = pd.read_csv(OUT / "image_manifest_syn.csv")
    paths = sorted(set(real.path) | set(syn.path))
    dev = "cuda"
    model = torch.hub.load(str(HUB), "dinov2_vitb14", source="local", pretrained=True).eval().to(dev)
    tf = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    feats = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(dev)
            f = model(x).float()
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu().numpy().astype(np.float16))
            if i % (bs * 20) == 0:
                print(i, len(paths), flush=True)
    F = np.concatenate(feats)
    np.savez_compressed(OUT / "features.npz", paths=np.array(paths), feats=F,
                        model=np.array("dinov2_vitb14 CLS, Resize256+CenterCrop224, L2-norm"))
    print("saved", F.shape, "device", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
