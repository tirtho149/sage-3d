"""Named Stage-1 training sets, built only from manifests (no results). Every set has a fixed image count per parent.

make(name) -> DataFrame(path, class_id, asset_id, root_sha1, arm, view, cond, edit_seed)
  "<ARM>"                         Stage-0R: V8 x {R0,R1,R2,R5}, edit seed 0 -> 32 images/parent (2D: 8 variants)
  "S0:<ARM>"                      original Stage-0 rows (audited record)
  "MIX<p>"                        p% KB3D / (100-p)% KB2D, 32 images/parent: views split deterministically per parent
  "<ARM>@<COND>"                  background ablation: 8 views x 1 condition = 8 images/parent (equal across conditions)
  "<ARM>#<VIEWSET>"               viewpoint ablation (NARROW / V8 / WIDE / INPUT), 8 views x 4 conditions
"""
import os
import zlib
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.generation.s1_const import VIEWSETS, ARMS3D, vkey  # noqa: E402
from src.s1.harness import manifest  # noqa: E402

COLS = ["path", "class_id", "asset_id", "root_sha1", "arm", "view", "cond", "edit_seed"]


def _v8(arm, seed=0, viewset="V8", conds=("R0", "R1", "R2", "R5")):
    d = manifest(arm); d = d[d.edit_seed.astype(int) == seed] if "edit_seed" in d else d
    if arm in ARMS3D:
        keep = {vkey(y, p) for (y, p) in VIEWSETS[viewset]}
    else:
        keep = {f"v{i}" for i in range(8)}
    d = d[d.view.isin(keep) & d.cond.isin(conds)]
    return d.drop_duplicates(["asset_id", "view", "cond"])[COLS]


DIV_EXCLUDE = {"coffee::brown_eye_spot", "mango::anthracnose"}   # < 12 parents reachable (photos < 256 px)


def diversity(arm, n_par, draw, nv=None):
    """Parent-diversity set: per class a seeded random order of its 12 parents (3 Stage-0 + 9 Stage-1), first n_par used;
    96 images/class: n_par=3 -> 8 variants x 4 conds, 6 -> v0-v3 x 4, 12 -> v0-v1 x 4. Nested across n_par for a draw."""
    d = manifest(arm); d = d[d.edit_seed.astype(int) == 0]
    d = d[~d.class_id.isin(DIV_EXCLUDE)]
    nv = nv or {3: 8, 6: 4, 12: 2}[n_par]
    out = []
    for c, g in d.groupby("class_id"):
        roots = sorted(g.root_sha1.unique())
        if len(roots) < 12:
            raise ValueError(f"{c}: {len(roots)} parents")
        order = np.random.default_rng(zlib.crc32(f"{c}|{draw}".encode())).permutation(roots)[:n_par]
        out.append(g[g.root_sha1.isin(order) & g.view.isin([f"v{i}" for i in range(nv)])])
    return pd.concat(out).drop_duplicates(["asset_id", "view", "cond"])[COLS]


def make(name, rep=0):
    """rep = replicate (edit seed / nuisance draw); arms are matched within a replicate."""
    if name.startswith("DIVR:"):                      # DIVR:<ARM>:P<n>:v<views>:d<draw>  (renders at fixed parents)
        _, arm, p, v, dr = name.split(":")
        return diversity(arm, int(p[1:]), int(dr[1:]), int(v[1:]))
    if name.startswith("DIV:"):                       # DIV:<ARM>:P<n>:d<draw>
        _, arm, p, dr = name.split(":")
        return diversity(arm, int(p[1:]), int(dr[1:]))
    if name.startswith("S0:"):
        return manifest(name)[COLS]
    if name.startswith("MIX"):
        p = int(name[3:]); A, B = _v8("KB3D", rep), _v8("KB2D", rep)
        out = []
        for aid in sorted(set(A.asset_id) & set(B.asset_id)):
            a, b = A[A.asset_id == aid].sort_values(["view", "cond"]), B[B.asset_id == aid].sort_values(["view", "cond"])
            rng = np.random.default_rng(zlib.crc32(aid.encode()))
            n3 = int(round(len(a) * p / 100)); ia = rng.permutation(len(a))[:n3]; ib = rng.permutation(len(b))[:len(b) - n3]
            out += [a.iloc[ia], b.iloc[ib]]
        d = pd.concat(out); d["arm"] = name; return d[COLS]
    if "@" in name:
        arm, cond = name.split("@")
        return _v8(arm, rep, conds=(cond,))                            # 8 views x 1 condition = 8 images/parent
    if "#" in name:
        arm, vs = name.split("#")
        if vs == "INPUT":
            d = _v8(arm, rep, viewset="V8"); d = d[d.view == (vkey(0, 0) if arm in ARMS3D else "v0")]
            return pd.concat([d] * 8)[COLS]                        # input view only: 4 conditions, each repeated 8x
        return _v8(arm, rep, viewset=vs)
    return _v8(name, rep)
