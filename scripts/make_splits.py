"""Leakage-safe splits for the E5 crops (all SAGE classes of each crop).

1. exact-duplicate removal by SHA-1 of image bytes
2. near-duplicate clustering: 64-bit pHash, union of pairs with Hamming <= 6 (within crop)
3. grouped, class-stratified split by cluster: build 45 / opt 25 / val 15 / test 15
Outputs data/metadata/splits_<crop>.csv and data/metadata/splits.sha256
"""
import io, hashlib, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, imagehash
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
SHARDS = Path.home() / "Desktop/Research/data/raw/SAGE/data"
CROPS = sys.argv[1:] or ["Peach", "Tomato", "Corn", "Citrus", "Mango"]
FRACS = [("build", .45), ("opt", .25), ("val", .15), ("test", .15)]
HAM = 6


def _hash(b):
    try:
        return str(imagehash.phash(Image.open(io.BytesIO(b)).convert("RGB")))
    except Exception:
        return None


man = pd.read_parquet(ROOT / "data/metadata/sage_shard_manifest.parquet", columns=["crop", "disease", "filename", "source", "shard"])
man = man[man.crop.isin(CROPS)]
rows = []
for shard in sorted(man.shard.unique()):
    t = pq.read_table(SHARDS / shard, columns=["crop", "disease", "filename", "source", "image"]).to_pandas()
    t = t[t.crop.isin(CROPS)]
    bs = [r["bytes"] for r in t.image]
    with ProcessPoolExecutor(10) as ex:
        ph = list(ex.map(_hash, bs, chunksize=64))
    for (_, r), b, h in zip(t.iterrows(), bs, ph):
        rows.append(dict(crop=r.crop, disease=r.disease, filename=r.filename, source=r.source, shard=shard,
                         sha1=hashlib.sha1(b).hexdigest(), phash=h))
    print(shard, len(t), flush=True)
df = pd.DataFrame(rows)
n0 = len(df); df = df.drop_duplicates("sha1"); df = df[df.phash.notna()]
print("rows", n0, "after exact dedup", len(df))


def clusters(hs):
    # union-find over Hamming<=HAM pairs, bucketed by 4 x 16-bit bands (pigeonhole: any pair within 6 bits shares >=1 band... not guaranteed, so use 8 x 8-bit bands)
    ints = np.array([int(h, 16) for h in hs], dtype=np.uint64)
    parent = np.arange(len(ints))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for band in range(8):   # 8 bands of 8 bits; Hamming<=6 => at least 2 bands identical (pigeonhole)
        key = (ints >> np.uint64(band * 8)) & np.uint64(0xFF)
        for k in np.unique(key):
            idx = np.where(key == k)[0]
            if len(idx) < 2 or len(idx) > 5000:
                continue
            sub = ints[idx]
            x = sub[:, None] ^ sub[None, :]
            d = np.unpackbits(x.view(np.uint8).reshape(len(idx), len(idx), 8), axis=2).sum(2)
            ii, jj = np.where(np.triu(d <= HAM, 1))
            for a, b in zip(idx[ii], idx[jj]):
                ra, rb = find(a), find(b)
                if ra != rb: parent[rb] = ra
    return np.array([find(i) for i in range(len(ints))])


out = []
rng = np.random.default_rng(0)
for crop, g in df.groupby("crop"):
    g = g.copy(); g["cluster"] = [f"{crop}:{c}" for c in clusters(g.phash.tolist())]
    # assign whole clusters, stratified by the cluster's majority class
    cl = g.groupby("cluster").agg(disease=("disease", lambda s: s.mode()[0]), n=("sha1", "size")).reset_index()
    cl["split"] = ""
    for dis, c in cl.groupby("disease"):
        c = c.sample(frac=1, random_state=int(rng.integers(1 << 31)))
        tot, cum, lab = c.n.sum(), 0, []
        for n in c.n:
            f = cum / tot; cum += n
            acc = 0
            for name, fr in FRACS:
                acc += fr
                if f < acc - 1e-9: lab.append(name); break
            else: lab.append("test")
        cl.loc[c.index, "split"] = lab
    g = g.merge(cl[["cluster", "split"]], on="cluster")
    g.drop(columns=[]).to_csv(ROOT / f"data/metadata/splits_{crop}.csv", index=False)
    out.append(g)
    print(crop, len(g), "clusters", g.cluster.nunique(), g.split.value_counts().to_dict(), flush=True)
allg = pd.concat(out)
h = hashlib.sha256(allg.sort_values("sha1")[["sha1", "split"]].to_csv(index=False).encode()).hexdigest()
open(ROOT / "data/metadata/splits.sha256", "w").write(h + "\n")
print("split hash", h)
