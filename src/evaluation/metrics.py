"""Image-level metrics for E1/E2/E4. All functions take uint8 HxWx3 arrays and bool masks."""
from __future__ import annotations
import numpy as np
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity

_lpips = None


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def delta_e_in_mask(x: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
    """Mean CIE76 ΔE inside mask."""
    m = m.astype(bool)
    if not m.any():
        return float("nan")
    return float(np.linalg.norm(rgb2lab(x)[m] - rgb2lab(y)[m], axis=-1).mean())


def ssim_in_mask(x: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
    _, S = structural_similarity(x, y, channel_axis=2, full=True, data_range=255)
    m = m.astype(bool)
    return float(S.mean(-1)[m].mean()) if m.any() else float("nan")


def lpips_masked(x: np.ndarray, y: np.ndarray, m: np.ndarray | None = None, device: str | None = None) -> float:
    """LPIPS(alex) on images with outside-mask pixels set to grey in both (so only the masked region differs)."""
    import os, torch, lpips
    device = device or os.environ.get("CLIP_DEVICE", "cuda")
    global _lpips
    if _lpips is None:
        _lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    if m is not None:
        x, y = x.copy(), y.copy(); x[~m.astype(bool)] = 128; y[~m.astype(bool)] = 128
    t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)
    with torch.no_grad():
        return float(_lpips(t(x), t(y)).item())


def preservation(before: np.ndarray, after: np.ndarray, organ: np.ndarray) -> dict:
    """Edit locality: change inside the target organ vs outside it (outside should be ~0)."""
    organ = organ.astype(bool)
    d = np.abs(before.astype(float) - after.astype(float)).mean(-1)
    return {"change_in": float(d[organ].mean()) if organ.any() else float("nan"),
            "change_out": float(d[~organ].mean()) if (~organ).any() else float("nan")}
