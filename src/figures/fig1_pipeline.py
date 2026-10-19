"""Figure 1 - Pipeline schematic (vector). Boxes are coloured by data status: real / synthetic / evaluation / blocked."""
import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from figures.common import TW, save

COL = dict(real="#d9ead3", syn="#fce5cd", kb="#cfe2f3", ev="#ead1dc", blk="#eeeeee")
W_, G_ = 0.150, 0.020
X = [i * (W_ + G_) for i in range(6)]
BOX = [  # (x, y, w, h, text, kind)
    (X[0], 0.60, W_, 0.28, "Real SAGE image\n(build split;\ndedup + pHash-safe)", "real"),
    (X[1], 0.60, W_, 0.28, "SAM ViT-H\norgan mask\n(box prompt)", "real"),
    (X[2], 0.60, W_, 0.28, "SAM 3D Objects\nGaussians + pose\n(+ pose fix A3*)", "syn"),
    (X[3], 0.60, W_, 0.28, "Geometry editor G2\nKB operators -> h\n+ ARAP, Laplacian,\nanchor  [J; generic G]", "syn"),
    (X[4], 0.60, W_, 0.28, "GaussianEditor\nappearance pass\n(geometry frozen)\n[K;  H = J -> K]", "syn"),
    (X[5], 0.60, W_, 0.28, "Fixed 18-view rig\n(same renderer\nfor every arm)\nSYNTHETIC, unverified", "syn"),
    (X[1], 0.08, 2 * W_ + G_, 0.32, "SAGE KB (verbatim web quote)\n-> stage segmentation -> structured spec:\noperators, appearance terms, prohibited,\nuncertain (hedged), citations", "kb"),
    (X[3], 0.08, W_ + 0.06, 0.32, "Baselines: A real; B, C,\nC-NG, C-MM, C-stage,\nI (2D shape-capable);\nD unedited; E/F GE;\nBmv/Cmv per-view 2D", "blk"),
    (X[4] + 0.07, 0.08, 1 - X[4] - 0.07, 0.32, "Evaluation: QC (grain, fuzz,\nNaN, collapse); geometry attrs;\nKB recall / unsupported;\ncross-view; specimen-level stats\n(expert review: BLOCKED)", "ev"),
]
ARROWS = [((X[i] + W_, 0.74), (X[i + 1], 0.74)) for i in range(5)] + [
    ((X[1] + W_ + G_ / 2, 0.40), (X[3] + W_ / 2, 0.60)), ((X[1] + W_ + G_ / 2, 0.40), (X[4] + W_ / 2, 0.60)),
    ((X[5] + W_ / 2, 0.60), (X[5] + W_ / 2, 0.40)), ((X[3] + W_ + 0.06, 0.24), (X[4] + 0.07, 0.24))]


def main():
    fig, ax = plt.subplots(figsize=(TW, 2.3)); ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1.0); ax.axis("off")
    for x, y, w, h, t, k in BOX:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.005,rounding_size=0.01", fc=COL[k], ec="#555", lw=0.5))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=4.4, linespacing=1.15)
    for (x0, y0), (x1, y1) in ARROWS:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#333", shrinkA=0, shrinkB=0))
    for i, (k, lab) in enumerate([("real", "real data"), ("syn", "synthetic (generated)"), ("kb", "source-grounded text"), ("ev", "evaluation"), ("blk", "baselines")]):
        ax.add_patch(FancyBboxPatch((0.0 + i * 0.2, 0.955), 0.018, 0.035, boxstyle="square,pad=0", fc=COL[k], ec="#555", lw=0.4))
        ax.text(0.024 + i * 0.2, 0.972, lab, va="center", fontsize=4.6)
    save(fig, "fig1_pipeline", ["src/**"], "Pipeline. Only the input image and mask are real; every 3D asset and render is synthetic. "
         "KB text enters only as verbatim cited phrases mapped to measurable operators/attributes.", vector=True)


if __name__ == "__main__":
    main()
