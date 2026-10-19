"""Shared figure utilities: one style, 600-dpi rasters, vector PDF for charts, provenance sidecars."""
import os, json, time
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/figures/paper"; DPI = 600; TW = 6.9
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "axes.titlesize": 6, "axes.labelsize": 6,
                     "xtick.labelsize": 5, "ytick.labelsize": 5, "legend.fontsize": 5, "pdf.fonttype": 42})
SYN = dict(color="#b00000", fontsize=4.5, fontweight="bold")


def img(p):
    return np.asarray(Image.open(p).convert("RGB")) if p and os.path.exists(p) else None


def crop_white(a, pad=6, thr=250):
    m = (a < thr).any(-1); ys, xs = np.nonzero(m)
    if len(ys) == 0: return a
    return a[max(0, ys.min() - pad):ys.max() + pad, max(0, xs.min() - pad):xs.max() + pad]


def show(ax, a, title=None, synthetic=False, note=None):
    ax.axis("off")
    if a is None:
        ax.text(0.5, 0.5, "not available", ha="center", va="center", color="gray", transform=ax.transAxes, fontsize=5)
    else:
        ax.imshow(a, interpolation="lanczos")
    if title: ax.set_title(title, pad=1.5)
    if synthetic: ax.text(0.02, 0.02, "SYNTHETIC\nunverified", transform=ax.transAxes, va="bottom", **SYN)
    if note: ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=4.5)


def save(fig, name, sources, caption, vector=False):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    if vector: fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    json.dump(dict(figure=name, generated=time.strftime("%Y-%m-%d %H:%M:%S"), sources=sources, caption=caption),
              open(f"{OUT}/{name}.json", "w"), indent=1)
