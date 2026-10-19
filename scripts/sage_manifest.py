"""Build a verified SAGE manifest from the HF image shards (non-image columns only).

Reads every data/train-*.parquet shard via HTTP range requests, keeping all
columns except the image bytes, plus a flag for whether image bytes are present.
Output: data/metadata/sage_shard_manifest.parquet
"""
import os, sys
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
from huggingface_hub import HfFileSystem

REPO = "datasets/tirtho149/SAGE"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "metadata", "sage_shard_manifest.parquet")
fs = HfFileSystem(token=os.environ.get("HF_TOKEN"))
shards = sorted(fs.glob(f"{REPO}/data/train-*.parquet"))
print(len(shards), "shards", flush=True)
tables = []
for s in shards:
    with fs.open(s, "rb", block_size=8 << 20) as f:
        pf = pq.ParquetFile(f)
        cols = [n for n in pf.schema_arrow.names if n != "image"]
        if len(tables) == 0:
            print("schema:", pf.schema_arrow, flush=True)
        t = pf.read(columns=cols)
        # image presence from row-group statistics (no image bytes downloaded)
        md = pf.metadata
        idx = [i for i in range(md.num_columns) if md.schema.column(i).path.startswith("image")]
        nulls = []
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(idx[0]).statistics
            nulls.append(st.null_count if st is not None and st.has_null_count else -1)
        print("  image column(s):", [md.schema.column(i).path for i in idx], "row-group null counts:", nulls, flush=True)
        t = t.append_column("shard", pa.array([os.path.basename(s)] * t.num_rows))
    tables.append(t)
    print(os.path.basename(s), t.num_rows, flush=True)
full = pa.concat_tables(tables, promote_options="default")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pq.write_table(full, OUT)
print("rows", full.num_rows, "->", OUT)
