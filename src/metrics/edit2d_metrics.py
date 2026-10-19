"""2D edit metrics vs the real input image (A).
  dE_in            mean Lab distance inside the original organ mask (appearance change)
  outside_change   area of pixels OUTSIDE the original mask with dE > 10, / mask area  (silhouette/shape-change capacity)
  detail_ratio     Laplacian energy inside the eroded mask, edited / input (<1: surface relief/texture smoothed away)
  bg_change        mean dE outside a 2x-dilated mask (non-target preservation)
usage: edit2d_metrics.py <input.png> <mask.png> <edited.png> ...   (prints one JSON line per edited image)
"""
import sys, json
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.color import rgb2lab


def metrics(inp, mask, ed):
    a = np.asarray(Image.open(inp).convert("RGB")) / 255.0; b = np.asarray(Image.open(ed).convert("RGB").resize(a.shape[1::-1])) / 255.0
    m = np.asarray(Image.open(mask).convert("L")) > 127; dE = np.linalg.norm(rgb2lab(a) - rgb2lab(b), axis=-1)
    r = max(1, int(np.sqrt(m.sum() / np.pi))); far = ~ndimage.binary_dilation(m, iterations=r)
    lap = lambda x, mm: float(np.mean(ndimage.laplace(x.mean(-1))[mm] ** 2))
    er = ndimage.binary_erosion(m, iterations=3)
    return dict(dE_in=float(dE[m].mean()), outside_change=float(((dE > 10) & ~m).sum() / m.sum()), detail_ratio=lap(b, er) / max(lap(a, er), 1e-12),
                bg_change=float(dE[far].mean()) if far.any() else float("nan"))


if __name__ == "__main__":
    for e in sys.argv[3:]:
        print(json.dumps(dict(file=e, **{k: round(v, 4) for k, v in metrics(sys.argv[1], sys.argv[2], e).items()})))
