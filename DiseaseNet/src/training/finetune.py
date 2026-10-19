"""T3: full fine-tuning of DINOv2 ViT-S/14 + linear head on a synthetic arm (fixed schedule, fp32).

Strict synthetic-only: hyper-parameters fixed in advance (below), last epoch is evaluated, no checkpoint
selection, no real label used. Measured on TITAN X: ~72 img/s at batch 64, 5.6 GB.
  epochs 15, batch 64, AdamW (backbone 2e-5, head 1e-3, wd 0.05), 5% warmup + cosine, label smoothing 0,
  augmentation RandomResizedCrop(224, scale 0.6-1) + h-flip + light colour jitter.
usage: CUDA_VISIBLE_DEVICES=0 ~/Desktop/Research/.venv/bin/python -m src.training.finetune <arm> [--fold k] [--epochs N] [--seed s]
  --fold k: synthetic IID run (train on assets != k, test on asset k); default: train on all, test on real.
  -> results/E0/t3_<arm>_<fold|real>_s<seed>.json (+ .npz predictions)
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.evaluation import metrics as Mx  # noqa: E402

HUB = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
NORM = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
TRAIN_TF = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.1, 0.1, 0.1, 0.02), T.ToTensor(), NORM])
EVAL_TF = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), NORM])


class DS(Dataset):
    def __init__(self, paths, y, tf):
        self.p, self.y, self.tf = list(paths), np.asarray(y), tf
    def __len__(self):
        return len(self.p)
    def __getitem__(self, i):
        return self.tf(Image.open(os.path.join(ROOT, self.p[i])).convert("RGB")), int(self.y[i])


class Net(torch.nn.Module):
    def __init__(self, n):
        super().__init__()
        self.enc = torch.hub.load(str(HUB), "dinov2_vits14", source="local", pretrained=True)
        self.head = torch.nn.Linear(384, n)
    def forward(self, x):
        return self.head(self.enc(x))


@torch.no_grad()
def predict(model, paths, y, bs=128):
    model.eval(); L = []
    for x, _ in DataLoader(DS(paths, y, EVAL_TF), batch_size=bs, num_workers=4):
        L.append(torch.softmax(model(x.cuda()), 1).cpu().numpy())
    return np.concatenate(L)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("arm"); ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    classes = sorted(c["class_id"] for c in json.load(open(f"{ROOT}/configs/datasets/stage0_classes.json"))["classes"])
    cix = {c: i for i, c in enumerate(classes)}
    syn = pd.concat([pd.read_csv(f"{ROOT}/data/synthetic/stage0/{m}") for m in ("manifest_3d.csv", "manifest_2d.csv")
                     if os.path.exists(f"{ROOT}/data/synthetic/stage0/{m}")])
    S = syn[syn.arm == a.arm].copy(); S["y"] = S.class_id.map(cix)
    S["rank"] = S.groupby("class_id").asset_id.transform(lambda s: s.map({x: i for i, x in enumerate(sorted(s.unique()))}))
    if a.fold is None:
        tr = S; real = pd.read_csv(f"{ROOT}/data/real_splits/stage0_test.csv"); te_p, te_y, te_g = real.path, real.class_id.map(cix).values, real.group.values
    else:
        tr = S[S["rank"] != a.fold]; te = S[S["rank"] == a.fold]; te_p, te_y, te_g = te.path, te.y.values, te.asset_id.values
    model = Net(len(classes)).cuda()
    dl = DataLoader(DS(tr.path, tr.y, TRAIN_TF), batch_size=64, shuffle=True, num_workers=6, drop_last=True)
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": 2e-5}, {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.05)
    total = a.epochs * len(dl); warm = max(1, int(0.05 * total)); base = [g["lr"] for g in opt.param_groups]
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        model.train(); tl = 0.0
        for x, y in dl:
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            loss = torch.nn.functional.cross_entropy(model(x.cuda()), y.cuda())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1; tl += float(loss)
        print(f"epoch {ep} loss {tl / len(dl):.3f} ({time.time() - t0:.0f}s)", flush=True)
    P = predict(model, te_p, te_y); pr = P.argmax(1)
    rs = Mx.summary(te_y, pr, P, classes=list(range(len(classes))))
    rs["macro_f1_ci"] = Mx.group_bootstrap(te_y, pr, te_g, lambda u, v: Mx.summary(u, v)["macro_f1"])
    tag = "real" if a.fold is None else f"fold{a.fold}"
    out = dict(arm=a.arm, eval=tag, seed=a.seed, epochs=a.epochs, n_train=int(len(tr)), metrics=rs,
               crop=Mx.crop_split(te_y, pr, classes), per_class_recall=dict(zip(classes, Mx.per_class_recall(te_y, pr, len(classes)))),
               train_sec=round(time.time() - t0, 1), schedule="fixed (no selection)")
    os.makedirs(f"{ROOT}/results/E0", exist_ok=True)
    json.dump(out, open(f"{ROOT}/results/E0/t3_{a.arm}_{tag}_s{a.seed}.json", "w"), indent=1)
    np.savez(f"{ROOT}/results/E0/t3_{a.arm}_{tag}_s{a.seed}.npz", y=te_y, proba=P)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rs.items()}), flush=True)


if __name__ == "__main__":
    main()
