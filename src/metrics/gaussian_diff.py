"""Per-Gaussian before/after comparison of two splats (e.g. SAM 3D asset D vs GaussianEditor output E/F).

Correspondence: GaussianEditor prunes (removes rows, order of survivors kept) and densifies (appends clones/splits).
We match each edited Gaussian to its nearest original by position (KD-tree) and accept a match only if it is
(i) mutual nearest neighbour and (ii) consistent with order preservation (matched original indices increase
along the edited array, checked via longest increasing run statistics). Unmatched edited rows = added,
unmatched original rows = deleted. Displacement is reported normalised by the original bbox diagonal.

usage (sam3d env): gaussian_diff.py <orig.ply> <edited.ply> <out.json>
"""
import sys, json
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

C0 = 0.28209479177387814


def load(p):
    v = PlyData.read(p)["vertex"].data
    g = lambda *n: np.stack([v[k] for k in n], 1).astype(np.float64)
    q = g("rot_0", "rot_1", "rot_2", "rot_3"); q /= np.linalg.norm(q, axis=1, keepdims=True)
    return dict(xyz=g("x", "y", "z"), rgb=np.clip(0.5 + C0 * g("f_dc_0", "f_dc_1", "f_dc_2"), 0, 1),
                op=1 / (1 + np.exp(-v["opacity"].astype(np.float64))), logs=g("scale_0", "scale_1", "scale_2"), q=q)


def match(A, B, block=64, max_shift=6):
    """Index correspondence, using GaussianEditor's verified behaviour (Peach 0006: output == input rows with
    sigmoid(opacity) >= 0.005, order kept): prune = row deletion with order preserved, densify = rows appended.
    1) drop A rows that the first prune removes (opacity < 0.005) -> candidate survivors A_k (original indices kept);
    2) align B to A_k block-wise: for each block of `block` B rows choose the offset (relative to the previous block,
       within +-max_shift) that minimises summed position distance. Residual off-by-a-few rows near shift points are
       possible; they inflate displacement statistics slightly (conservative for an "appearance-only" verdict).
    3) B rows beyond the aligned length = appended (densified) Gaussians."""
    keep = np.nonzero(A["op"] >= 0.005)[0]; xa, xb = A["xyz"][keep], B["xyz"]
    nA, nB = len(xa), len(xb); ia, jb = [], []; off = 0
    for j0 in range(0, nB, block):
        j = np.arange(j0, min(j0 + block, nB)); best = None
        for d in range(-max_shift, max_shift + 1):
            i = j + off + d
            if off + d < 0 or i[-1] >= nA: continue
            c = np.linalg.norm(xa[i] - xb[j], axis=1).sum()
            if best is None or c < best[0]: best = (c, d)
        if best is None: break                       # ran past the survivors -> remaining B rows are appended
        off += best[1]; ia.extend(keep[j + off]); jb.extend(j)
    ia, jb = np.array(ia), np.array(jb)
    return ia, jb, float((np.diff(ia) == 1).mean()) if len(ia) > 1 else float("nan")


def stats(x):
    return dict(mean=float(x.mean()), median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(x.max()))


def diff(orig, edited):
    A, B = load(orig), load(edited); ia, jb, order_ok = match(A, B)
    diag = float(np.linalg.norm(A["xyz"].max(0) - A["xyz"].min(0)))
    disp = np.linalg.norm(B["xyz"][jb] - A["xyz"][ia], axis=1) / diag
    dq = np.abs((A["q"][ia] * B["q"][jb]).sum(1)).clip(0, 1); rot = np.degrees(2 * np.arccos(dq))
    dls = np.abs(B["logs"][jb] - A["logs"][ia]).mean(1)
    dop = B["op"][jb] - A["op"][ia]; drgb = np.linalg.norm(B["rgb"][jb] - A["rgb"][ia], axis=1)
    # coherence: is the displacement a smooth deformation field or per-Gaussian jitter?
    # ratio = |mean of neighbours' displacement vectors| / mean |displacement| over k-NN (in the original);
    # i.i.d. jitter -> ~1/sqrt(k) (=0.25 for k=16); rigid/smooth deformation -> ~1.
    rng = np.random.default_rng(0); sub = rng.choice(len(ia), min(20000, len(ia)), replace=False)
    D = B["xyz"][jb] - A["xyz"][ia]; tr = cKDTree(A["xyz"][ia]); _, nn = tr.query(A["xyz"][ia][sub], k=16)
    coh = np.linalg.norm(D[nn].mean(1), axis=1) / (np.linalg.norm(D[nn], axis=2).mean(1) + 1e-12)
    # thickness: local PCA smallest std over 32-NN, relative to bbox diag (a leaf is a thin sheet; jitter thickens it)
    def thick(X, idx):
        _, nb = cKDTree(X).query(X[idx], k=32); P = X[nb] - X[nb].mean(1, keepdims=True)
        return np.sqrt(np.linalg.eigvalsh(np.einsum("nki,nkj->nij", P, P) / 32)[:, 0]) / diag
    ta = thick(A["xyz"], rng.choice(len(A["xyz"]), 20000, replace=False)); tb = thick(B["xyz"], rng.choice(len(B["xyz"]), 20000, replace=False))
    # silhouette-free geometry summary: bbox and spread change of all Gaussians (incl. added)
    ext = lambda X: X.max(0) - X.min(0)
    return dict(
        n_orig=len(A["xyz"]), n_edit=len(B["xyz"]), n_matched=int(len(jb)), n_added=int(len(B["xyz"]) - len(jb)),
        n_deleted=int(len(A["xyz"]) - len(ia)), consecutive_match_frac=order_ok, bbox_diag=diag,
        disp_rel=stats(disp), frac_disp_gt_1pct=float((disp > 0.01).mean()), rot_deg=stats(rot), dlogscale=stats(dls),
        dopacity=dict(stats(np.abs(dop)), mean_signed=float(dop.mean())), drgb=stats(drgb),
        disp_coherence_k16=dict(median=float(np.median(coh)), iid_reference=0.25),
        sheet_thickness_rel=dict(orig_median=float(np.median(ta)), edit_median=float(np.median(tb)), ratio=float(np.median(tb) / np.median(ta))),
        bbox_extent_ratio=(ext(B["xyz"]) / ext(A["xyz"])).tolist(),
        mean_opacity=[float(A["op"].mean()), float(B["op"].mean())],
        classification=("no change" if np.percentile(disp, 95) < 1e-6 and np.percentile(drgb, 95) < 1e-6 else
                        "appearance-only" if np.percentile(disp, 95) < 0.01 else
                        "geometric jitter (incoherent)" if np.median(coh) < 0.5 else "coherent geometric deformation"),
    ), dict(ia=ia, jb=jb, disp=disp, drgb=drgb)


if __name__ == "__main__":
    r, _ = diff(sys.argv[1], sys.argv[2]); json.dump(r, open(sys.argv[3], "w"), indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)}))
    for k in ("disp_rel", "rot_deg", "dlogscale", "dopacity", "drgb", "disp_coherence_k16", "sheet_thickness_rel"): print(k, {a: round(b, 4) for a, b in r[k].items()})
