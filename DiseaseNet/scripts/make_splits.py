"""Phase 0 / Part D: locked, leakage-safe real SAGE splits for every eligible class.

Done BEFORE any synthetic image exists. Steps
  1. pHash (64-bit) of every unique image in the eligible classes (tier B/C), from the shards.
  2. Near-duplicate groups across ALL eligible classes: union-find over pairs with Hamming
     distance <= HAM. Candidates come from 8 bands of 8 bits; by pigeonhole any pair within
     distance <= 7 shares an identical band, so no pair within HAM is missed.
     Exact duplicates (same sha1) are the same node. A group is the split unit.
  3. Open-set classes: OPEN_FRAC of eligible classes (seeded, stratified by tier, excluding
     the 5 classes that already have 3D assets) are reserved entirely -> partition "openset".
  4. Within each known class, groups are assigned (seeded) to
        parent  : source photos allowed for 3D reconstruction / 2D edits / backgrounds
        adapt   : optional real adaptation pool (few-shot, H9 / L2)
        val     : real validation, only for the declared real-assisted setting
        test    : LOCKED final test
     A group that spans several classes goes to the same partition everywhere (by group id).
  5. Provenance: the source photos of the 15 existing SAM 3D assets are forced into parent.
  6. A lock file records counts and the sha256 of the sorted test sha1 list; assertions in
     src/provenance/leakage.py check every later derived image against it.
Caveat (protocol D1): SAGE has no reliable specimen ids, so pHash-disjoint does not prove
specimen independence; same-specimen grouping is unresolved where evidence is missing.

usage: ~/Desktop/Research/.venv/bin/python scripts/make_splits.py
"""
import hashlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor

import imagehash
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAGE = os.path.expanduser("~/Desktop/Research/data/raw/SAGE/data")
EXT = os.path.expanduser("~/Desktop/Research/external/SAGE-Extend")
HAM = 6
OPEN_FRAC = 0.10
FRACS = dict(parent=0.40, adapt=0.15, val=0.15, test=0.30)
SEED = 20261015
PILOT_CLASSES = ["peach::leaf_curl", "tomato::leaf_mold", "corn::smut", "citrus::canker", "mango::anthracnose"]


def phash_shard(args):
    shard, rows = args
    pf = pq.ParquetFile(f"{SAGE}/{shard}"); out = {}
    for rg, idx in rows.items():
        col = pf.read_row_group(rg, columns=["image"]).column("image").to_pylist()
        for i, sha in idx:
            if sha in out:
                continue
            try:
                im = Image.open(io.BytesIO(col[i]["bytes"])).convert("RGB")
                out[sha] = int(str(imagehash.phash(im)), 16)
            except Exception:
                out[sha] = None
    return out


class UF:
    def __init__(self, n): self.p = np.arange(n)
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.p[max(a, b)] = min(a, b)


def popcount64(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((x * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.int64)


def main():
    os.makedirs(f"{ROOT}/data/real_splits", exist_ok=True)
    C = pd.read_parquet(f"{ROOT}/metadata/class_ontology.parquet")
    elig = C[C.tier.isin(["B", "C"])]
    M = pd.read_parquet(f"{ROOT}/data/manifests/sage_images.parquet",
                        columns=["class_id", "sha1", "shard", "row_group", "row_in_group", "source", "filename"])
    M = M[M.class_id.isin(elig.class_id) & M.sha1.notna()]
    # 1. pHash per unique sha1
    first = M.drop_duplicates("sha1")
    jobs = []
    for shard, g in first.groupby("shard"):
        rows = {}
        for rg, h in g.groupby("row_group"):
            rows[int(rg)] = list(zip(h.row_in_group.astype(int), h.sha1))
        jobs.append((shard, rows))
    ph = {}
    with ProcessPoolExecutor(10) as ex:
        for d in ex.map(phash_shard, jobs):
            ph.update(d)
    U = pd.DataFrame({"sha1": list(ph), "phash": list(ph.values())}).dropna()
    U["phash"] = U.phash.astype("uint64")
    # 2. near-duplicate groups (band candidates, exact Hamming check)
    H = U.phash.values.astype(np.uint64); n = len(H); uf = UF(n)
    for b in range(8):
        band = ((H >> np.uint64(8 * b)) & np.uint64(0xFF)).astype(np.int64)
        order = np.argsort(band, kind="stable"); bs = band[order]
        cuts = np.flatnonzero(np.diff(bs)) + 1
        for grp in np.split(order, cuts):
            if len(grp) < 2: continue
            x = H[grp]
            for k in range(len(grp) - 1):
                d = popcount64(x[k] ^ x[k + 1:])
                for j in np.flatnonzero(d <= HAM):
                    uf.union(grp[k], grp[k + 1 + j])
    U["group"] = [uf.find(i) for i in range(n)]
    M = M.merge(U[["sha1", "phash", "group"]], on="sha1", how="inner")
    # 3. open-set classes
    rng = np.random.default_rng(SEED)
    cand = elig[~elig.class_id.isin(PILOT_CLASSES)]
    open_cls = []
    for tier, g in cand.groupby("tier"):
        k = int(round(OPEN_FRAC * len(g)))
        open_cls += list(rng.choice(g.class_id.values, k, replace=False))
    # 4/5. group -> partition (one partition per group, everywhere)
    spec = pd.read_csv(f"{EXT}/data/metadata/specimen_manifest.csv")
    assets = sorted(d for d in os.listdir(f"{EXT}/results/pilot/sam3d") if "__" in d) if os.path.isdir(f"{EXT}/results/pilot/sam3d") else []
    asset_ids = {a.rsplit("__", 1)[0] + "/" + a.rsplit("__", 1)[1] for a in assets}
    asset_sha = set(spec[spec.specimen_id.isin(asset_ids)].sha1)
    forced_parent = set(M[M.sha1.isin(asset_sha)].group)
    groups = M.group.unique(); rng.shuffle(groups)
    names = list(FRACS); cum = np.cumsum([FRACS[k] for k in names])
    part = {}
    for i, gid in enumerate(groups):
        u = (i + 0.5) / len(groups)
        part[gid] = "parent" if gid in forced_parent else names[int(np.searchsorted(cum, u))]
    M["partition"] = M.group.map(part)
    M.loc[M.class_id.isin(open_cls), "partition"] = "openset"
    # a group touching an open-set class must not appear in known-class train pools either
    open_groups = set(M[M.class_id.isin(open_cls)].group)
    M.loc[M.group.isin(open_groups) & ~M.class_id.isin(open_cls) & M.partition.isin(["parent", "adapt"]), "partition"] = "excluded_openset_overlap"
    M.to_parquet(f"{ROOT}/data/real_splits/real_splits.parquet", index=False)
    # 6. lock
    test = sorted(M[M.partition == "test"].sha1.unique())
    lock = dict(created="2026-10-15", seed=SEED, ham_threshold=HAM, open_frac=OPEN_FRAC, fracs=FRACS,
                n_unique_images=int(M.sha1.nunique()), n_groups=int(M.group.nunique()),
                partitions={k: int(v) for k, v in M.drop_duplicates("sha1").partition.value_counts().items()},
                n_known_classes=int(len(elig) - len(open_cls)), open_set_classes=sorted(open_cls),
                asset_source_images_forced_parent=int(len(asset_sha & set(M.sha1))),
                test_sha1_sha256=hashlib.sha256("\n".join(test).encode()).hexdigest())
    json.dump(lock, open(f"{ROOT}/data/real_splits/LOCK.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in lock.items() if k != "open_set_classes"}, indent=1))


if __name__ == "__main__":
    main()
