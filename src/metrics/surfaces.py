"""Surface-placement measures for upper vs lower leaf symptoms (Tomato Leaf Mold demo).

Per rendered pixel we know which anatomical surface is visible:
  * bilayer assets: render the layer id as colour (upper red, lower blue);
  * single-sheet assets (D, GaussianEditor, ...): a Gaussian shows its ADAXIAL side to a camera when its adaxial
    normal points toward that camera (n . (c - mu) > 0); rendered as colour per view.
Colour classes (CIELAB on the rendered/edited RGB; thresholds fixed before looking at edited outputs and applied to
every method alike; they test WHERE a colour lands, not whether it is biologically right):
  mold   'olive-green to brown': L* < 52, hue angle 55-108 deg, chroma > 12, AND darker than the lamina median by > 6
  spot   'pale greenish-yellow': L* above the lamina median by > 8 and hue angle < lamina median hue - 3
Outputs per method: fraction of upper-surface pixels classed mold (wrong surface for the KB), fraction of
lower-surface pixels classed mold, fraction of lower-surface pixels classed spot (the photo's upper spots showing
through on the underside: the single-sheet mirror error).
"""
import numpy as np
from plyfile import PlyData


def lab(rgb8):
    from geometry_editor.anatomy import srgb_to_lab
    return srgb_to_lab(rgb8.reshape(-1, 3) / 255.0).reshape(rgb8.shape)


def classes(rgb8, mask, ref=None):
    """ref: (L_median, hue_median) of healthy-looking lamina; computed from `mask` pixels if None."""
    Lb = lab(rgb8); L, A, B = Lb[..., 0], Lb[..., 1], Lb[..., 2]; hue = np.degrees(np.arctan2(B, A)); ch = np.hypot(A, B)
    if ref is None: ref = (float(np.median(L[mask])), float(np.median(hue[mask])))
    mold = mask & (L < 52) & (hue > 55) & (hue < 108) & (ch > 12) & (L < ref[0] - 6)
    spot = mask & (L > ref[0] + 8) & (hue < ref[1] - 3)
    return mold, spot, ref


def camera_centres(rig):
    return {c["id"]: -np.asarray(c["R"]).T @ np.asarray(c["t"]) for c in rig["views"]}


def side_colours_single(X, n_adax, cam):
    s = ((cam[None] - X) * n_adax).sum(1) > 0
    c = np.zeros((len(X), 3)); c[s] = [1, 0, 0]; c[~s] = [0, 0, 1]; return c


def surface_stats(renders, side_renders, ref):
    """renders/side_renders: lists of rgb uint8 (same views). Returns pooled fractions over all views."""
    up = lo = up_m = lo_m = lo_s = 0
    for rgb8, sd in zip(renders, side_renders):
        sd = np.asarray(sd).astype(np.int32)                         # uint8 + 40 would overflow (255 + 40 -> 39)
        fg = (rgb8.astype(int).sum(-1) < 735)                       # non-white
        is_up = fg & (sd[..., 0] > sd[..., 2] + 40); is_lo = fg & (sd[..., 2] > sd[..., 0] + 40)
        mold, spot, _ = classes(rgb8, fg, ref)
        up += is_up.sum(); lo += is_lo.sum(); up_m += (mold & is_up).sum(); lo_m += (mold & is_lo).sum(); lo_s += (spot & is_lo).sum()
    return dict(upper_px=int(up), lower_px=int(lo), mold_on_upper=float(up_m / max(up, 1)), mold_on_lower=float(lo_m / max(lo, 1)),
                spot_on_lower=float(lo_s / max(lo, 1)))
