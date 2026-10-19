"""Stage 1 classifier fine-tuning study (T1 linear head, T2 last-k blocks, T3 LoRA, T4 full) on DINOv2 ViT-S/14.

Modes
  --mode linear | lastk --k 1|2 | lora --r 8 | full
Options: --lr-head, --lr-bb (backbone/LoRA), --llrd (layer-wise decay, full/lastk), --wd, --ls (label smoothing),
         --balanced (class-balanced sampler), --epochs, --clip (grad-norm), --aug std|strong, --seed
Validation (selection) — strict S0 uses only synthetic data:
  --val-fold k : hold out every class's parent of rank k (LOPO); log val loss / macro-F1 each epoch; report the best
                 epoch (no DEV involvement). Used for hyper-parameter search and to pick the number of epochs.
  no --val-fold: train on all parents for --epochs (chosen beforehand by LOPO), evaluate DEV (and FINAL if --final).
Every run logs train loss, val loss/F1 per epoch, and embedding drift = 1 - mean cosine between the pretrained and
the fine-tuned CLS embedding on a fixed set of 512 DEV photos (no labels used).
Outputs results/stage1/ft/<tag>.json (+ .npz DEV/FINAL probabilities).
usage: CUDA_VISIBLE_DEVICES=0 ~/Desktop/Research/.venv/bin/python -m src.s1.finetune KB3D --mode lastk --k 2 --val-fold 0
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.evaluation import metrics as Mx  # noqa: E402
from src.s1 import harness as Hn  # noqa: E402
from src.s1.sets import make  # noqa: E402

HUB = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
NORM = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
AUG = {"std": T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip(),
                         T.ColorJitter(0.1, 0.1, 0.1, 0.02), T.ToTensor(), NORM]),
       "strong": T.Compose([T.RandomResizedCrop(224, scale=(0.35, 1.0), interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip(),
                            T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.8), T.RandomGrayscale(0.05),
                            T.RandomApply([T.GaussianBlur(9, (0.1, 2.0))], p=0.3), T.ToTensor(), NORM, T.RandomErasing(0.25)])}
EVAL_TF = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), NORM])


class DS(Dataset):
    def __init__(self, paths, y, tf):
        self.p, self.y, self.tf = list(paths), np.asarray(y), tf
    def __len__(self):
        return len(self.p)
    def __getitem__(self, i):
        return self.tf(Image.open(os.path.join(ROOT, self.p[i])).convert("RGB")), int(self.y[i])


def encoder():
    return torch.hub.load(str(HUB), "dinov2_vits14", source="local", pretrained=True)


class Net(torch.nn.Module):
    def __init__(self, n, mode, k=2, r=8):
        super().__init__()
        self.enc = encoder(); self.head = torch.nn.Linear(384, n); self.mode = mode
        for p in self.enc.parameters():
            p.requires_grad = False
        if mode == "lastk":
            for b in self.enc.blocks[-k:]:
                for p in b.parameters():
                    p.requires_grad = True
            for p in self.enc.norm.parameters():
                p.requires_grad = True
        elif mode == "full":
            for p in self.enc.parameters():
                p.requires_grad = True
        elif mode == "lora":
            from peft import LoraConfig, get_peft_model
            self.enc = get_peft_model(self.enc, LoraConfig(r=r, lora_alpha=2 * r, target_modules=["qkv"], lora_dropout=0.05))
    def forward(self, x):
        return self.head(self.enc(x))


def param_groups(model, a):
    groups = [{"params": list(model.head.parameters()), "lr": a.lr_head}]
    if a.mode == "linear":
        return groups
    if a.mode == "lora":
        return groups + [{"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": a.lr_bb}]
    blocks = model.enc.blocks; nb = len(blocks)
    for i, b in enumerate(blocks):
        ps = [p for p in b.parameters() if p.requires_grad]
        if ps:
            groups.append({"params": ps, "lr": a.lr_bb * (a.llrd ** (nb - 1 - i))})
    rest = [p for n, p in model.enc.named_parameters() if p.requires_grad and not n.startswith("blocks.")]
    if rest:
        groups.append({"params": rest, "lr": a.lr_bb})
    return groups


@torch.no_grad()
def predict(model, paths, bs=128):
    model.eval(); L = []
    for x, _ in DataLoader(DS(paths, np.zeros(len(paths)), EVAL_TF), batch_size=bs, num_workers=4):
        L.append(torch.softmax(model(x.cuda()), 1).float().cpu().numpy())
    return np.concatenate(L)


@torch.no_grad()
def embed(enc, paths, bs=128):
    enc.eval(); L = []
    for x, _ in DataLoader(DS(paths, np.zeros(len(paths)), EVAL_TF), batch_size=bs, num_workers=4):
        L.append(torch.nn.functional.normalize(enc(x.cuda()).float(), dim=1).cpu())
    return torch.cat(L)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("set")
    ap.add_argument("--mode", default="full"); ap.add_argument("--k", type=int, default=2); ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-3); ap.add_argument("--lr-bb", type=float, default=2e-5)
    ap.add_argument("--llrd", type=float, default=1.0); ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ls", type=float, default=0.0); ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--aug", default="std"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--val-fold", type=int, default=None); ap.add_argument("--final", default=None, help="final cohort name (frozen configs only)")
    ap.add_argument("--tag", default=None); ap.add_argument("--out", default="results/stage1/ft")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    classes = Hn.classes20(); cix = {c: i for i, c in enumerate(classes)}
    S = make(a.set); S = S[S.class_id.isin(classes)].copy(); S["y"] = S.class_id.map(cix)
    S["rank"] = S.groupby("class_id").root_sha1.transform(lambda s: s.map({x: i for i, x in enumerate(sorted(s.unique()))}))
    tr, va = (S[S["rank"] != a.val_fold], S[S["rank"] == a.val_fold]) if a.val_fold is not None else (S, None)
    tag = a.tag or f"{a.set}_{a.mode}{a.k if a.mode == 'lastk' else ''}_lr{a.lr_bb:g}_ls{a.ls:g}_{a.aug}{'_bal' if a.balanced else ''}_e{a.epochs}_s{a.seed}" + \
        (f"_fold{a.val_fold}" if a.val_fold is not None else "")
    out = f"{ROOT}/{a.out}"; os.makedirs(out, exist_ok=True)
    model = Net(len(classes), a.mode, a.k, a.r).cuda()
    sampler = None
    if a.balanced:
        w = 1.0 / tr.groupby("y").y.transform("size").values; sampler = WeightedRandomSampler(torch.tensor(w), len(tr), replacement=True)
    dl = DataLoader(DS(tr.path, tr.y, AUG[a.aug]), batch_size=a.bs, shuffle=sampler is None, sampler=sampler, num_workers=6, drop_last=True)
    opt = torch.optim.AdamW(param_groups(model, a), weight_decay=a.wd)
    total = a.epochs * len(dl); warm = max(1, int(0.05 * total)); base = [g["lr"] for g in opt.param_groups]
    lossf = torch.nn.CrossEntropyLoss(label_smoothing=a.ls)
    hist, best, step, t0 = [], (-1, None, None), 0, time.time()
    for ep in range(a.epochs):
        model.train(); tl, nb = 0.0, 0
        for x, y in dl:
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            loss = lossf(model(x.cuda()), y.cuda()); opt.zero_grad(set_to_none=True); loss.backward()
            if a.clip > 0:
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], a.clip)
            opt.step(); step += 1; tl += float(loss); nb += 1
        h = dict(epoch=ep, train_loss=tl / nb)
        if va is not None:
            P = predict(model, va.path); yv = va.y.values
            h["val_loss"] = float(-np.log(P[np.arange(len(yv)), yv] + 1e-9).mean())
            h["val_macro_f1"] = Mx.summary(yv, P.argmax(1), classes=list(range(len(classes))))["macro_f1"]
            if h["val_macro_f1"] > best[0]:
                best = (h["val_macro_f1"], ep, None)
        hist.append(h); print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in h.items()}), f"{time.time() - t0:.0f}s", flush=True)
    res = dict(set=a.set, args=vars(a), n_train=int(len(tr)), steps=step, history=hist, train_sec=round(time.time() - t0, 1),
               trainable_params=int(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    store = {}
    if va is not None:
        res["best_val_macro_f1"], res["best_epoch"] = best[0], best[1]; res["last_val_macro_f1"] = hist[-1]["val_macro_f1"]
    else:
        D = Hn.dev_set(); D = D[D.class_id.isin(classes)]
        P = predict(model, D.path); yy = D.class_id.map(cix).values
        res["dev"] = Hn.evaluate(yy, P, D.group.values, len(classes), classes); store.update(dev_proba=P, dev_y=yy, dev_g=D.group.values)
        if a.final:
            F = Hn.final_set(a.final, tag); F = F[F.class_id.isin(classes)]
            P = predict(model, F.path); yf = F.class_id.map(cix).values
            res["final"] = Hn.evaluate(yf, P, F.group.values, len(classes), classes); store.update(final_proba=P, final_y=yf, final_g=F.group.values)
        probe = D.sample(min(512, len(D)), random_state=0).path.tolist()
        e1 = embed(model.enc, probe); e0 = embed(encoder().cuda(), probe)
        res["embedding_drift"] = float(1 - (e0 * e1).sum(1).mean())
        print("DEV", json.dumps({k: round(v, 4) for k, v in res["dev"].items() if isinstance(v, float)}), "drift", round(res["embedding_drift"], 4), flush=True)
    json.dump(res, open(f"{out}/{tag}.json", "w"), indent=1)
    if store:
        np.savez_compressed(f"{out}/{tag}.npz", **store)


if __name__ == "__main__":
    main()
