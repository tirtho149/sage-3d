"""CLIP-based automatic metrics for E2/E4 (ViT-L/14, openai).

- direction_similarity: cos(E_img(edited)-E_img(orig), E_txt(target)-E_txt(source))  [IN2N/GaussCtrl metric]
- kb_specificity: cos(E_img(edited)-E_img(orig), E_txt(kb clause)-E_txt(src)) - same with mismatched-disease clause
- directional_consistency: mean cos between edit directions of ADJACENT views (rig neighbours)  [IN2N/GaussCtrl metric]
These are proxies. They do not establish biological correctness (needs expert raters).
"""
from __future__ import annotations
import os, numpy as np, torch
DEV = os.environ.get("CLIP_DEVICE", "cuda")
from PIL import Image

_M = None


def _model():
    global _M
    if _M is None:
        from transformers import CLIPModel, CLIPProcessor
        _M = (CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(DEV), CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14"))
    return _M


@torch.no_grad()
def img_emb(imgs):
    m, p = _model()
    x = p(images=[Image.fromarray(i) if isinstance(i, np.ndarray) else i for i in imgs], return_tensors="pt").to(DEV)
    e = m.get_image_features(**x); return torch.nn.functional.normalize(e, dim=-1)


@torch.no_grad()
def txt_emb(texts):
    m, p = _model()
    x = p(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(DEV)
    e = m.get_text_features(**x); return torch.nn.functional.normalize(e, dim=-1)


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def direction_similarity(orig, edited, src_text, tgt_text):
    di = img_emb(edited) - img_emb(orig); dt = txt_emb([tgt_text]) - txt_emb([src_text])
    return _cos(di, dt.expand_as(di)).cpu().numpy()


def kb_specificity(orig, edited, src_text, kb_text, mismatched_text):
    di = img_emb(edited) - img_emb(orig); t = txt_emb([src_text, kb_text, mismatched_text])
    return (_cos(di, (t[1] - t[0]).expand_as(di)) - _cos(di, (t[2] - t[0]).expand_as(di))).cpu().numpy()


def directional_consistency(orig_views, edited_views, neighbours):
    """neighbours: list of (i, j) adjacent view pairs on the rig."""
    d = img_emb(edited_views) - img_emb(orig_views)
    d = torch.nn.functional.normalize(d, dim=-1)
    return float(np.mean([_cos(d[i], d[j]).item() for i, j in neighbours]))


def rig_neighbours(n_az=12, n_rings=2):
    pairs = []
    for r in range(n_rings):
        for i in range(n_az):
            pairs.append((r * n_az + i, r * n_az + (i + 1) % n_az))
    for i in range(n_az):
        pairs.append((i, n_az + i))      # +20 vs -20 elevation, same azimuth
    return pairs
