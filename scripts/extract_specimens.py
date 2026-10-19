"""Extract images of the candidate classes from local SAGE shards.

Writes data/specimens/<crop>__<disease>/<idx>_<filename> and a manifest CSV.
Each SAGE image is treated as an independent specimen (SAGE has no specimen IDs);
exact-duplicate bytes are removed so one physical photo cannot appear twice.
"""
import sys, hashlib, io, re
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq, yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SHARDS = Path.home() / "Desktop/Research/data/raw/SAGE/data"
cfg = yaml.safe_load(open(ROOT / "experiments/configs/candidates.yaml"))
man = pd.read_parquet(ROOT / "data/metadata/sage_shard_manifest.parquet",
                      columns=["crop", "disease", "filename", "source", "shard"])
want = {(c["crop"], c["disease"]): c["category"] for c in cfg["classes"]}
sel = man[[k in want for k in zip(man.crop, man.disease)]].copy()
print(sel.groupby(["crop", "disease"]).size())
rows = []
for shard, g in sel.groupby("shard"):
    t = pq.read_table(SHARDS / shard, columns=["crop", "disease", "filename", "source", "image"]).to_pandas()
    t = t[[k in want for k in zip(t.crop, t.disease)]]
    for _, r in t.iterrows():
        b = r.image["bytes"]
        rows.append(dict(crop=r.crop, disease=r.disease, category=want[(r.crop, r.disease)], filename=r.filename,
                         source=r.source, shard=shard, sha1=hashlib.sha1(b).hexdigest(), img_bytes=b))
df = pd.DataFrame(rows).drop_duplicates("sha1")
out = []
for (crop, dis), g in df.groupby(["crop", "disease"]):
    g = g.sample(frac=1, random_state=cfg["seed"]).head(cfg["cap_per_class"])
    d = ROOT / "data/specimens" / re.sub(r"\W+", "_", f"{crop}__{dis}")
    d.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(g.itertuples()):
        im = Image.open(io.BytesIO(r.img_bytes)).convert("RGB")
        p = d / f"{i:04d}_{re.sub(r'[^A-Za-z0-9_.-]', '_', r.filename)}"
        p = p.with_suffix(".png"); im.save(p)
        out.append(dict(specimen_id=f"{d.name}/{i:04d}", crop=crop, disease=dis, category=r.category,
                        sage_filename=r.filename, sage_source=r.source, shard=r.shard, sha1=r.sha1,
                        width=im.width, height=im.height, path=str(p.relative_to(ROOT))))
pd.DataFrame(out).to_csv(ROOT / "data/metadata/specimen_manifest.csv", index=False)
print(pd.DataFrame(out).groupby(["category", "crop", "disease"]).size())
