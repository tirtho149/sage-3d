"""Stage 1: extract real SAGE photos of one locked partition for a class cohort into the local JPEG cache.

Cohorts: s0 = the 20 Stage-0 classes; open = the 19 open-set reserve classes (unknowns; groups shared with known classes removed); all = Stage-0 classes + every Stage-1-eligible class (CLASS_ELIGIBILITY.csv).
Seeded per-class cap (seed 20261016). The `val` partition is the Stage-1 FINAL test: extracting it is allowed
(no labels are looked at), evaluating on it is guarded in src/s1/harness.py.
usage: ~/Desktop/Research/.venv/bin/python scripts/s1_extract_real.py <partition> <cap> <cohort> <out_csv_name>
  -> data/real_splits/cache/<partition>/<sha1>.jpg, data/real_splits/<out_csv_name>.csv
"""
import io, json, os, sys
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src.provenance.leakage import Guard  # noqa: E402
SAGE = os.path.expanduser("~/Desktop/Research/data/raw/SAGE/data"); SEED = 20261016


def main(part, cap, cohort, name):
    Guard()
    s0 = [c["class_id"] for c in json.load(open(f"{ROOT}/configs/datasets/stage0_classes.json"))["classes"]]
    E = pd.read_csv(f"{ROOT}/CLASS_ELIGIBILITY.csv")
    lock = json.load(open(f"{ROOT}/data/real_splits/LOCK.json"))
    classes = (s0 if cohort == "s0" else lock["open_set_classes"] if cohort == "open"
               else sorted(set(s0) | set(E[E.stage1_eligible].class_id)))
    S = pd.read_parquet(f"{ROOT}/data/real_splits/real_splits.parquet")
    S = S[S.class_id.isin(classes) & (S.partition == part)].drop_duplicates(["sha1", "class_id"])
    S = S[~S.sha1.duplicated(keep=False)]                 # drop photos that carry >1 label within the cohort
    if cohort == "open":                                  # unknowns must not be near-duplicates of any known-class photo
        R = pd.read_parquet(f"{ROOT}/data/real_splits/real_splits.parquet", columns=["group", "partition"])
        known_groups = set(R[~R.partition.isin(["openset"])].group)
        S = S[~S.group.isin(known_groups)]
    sel = pd.concat([g.sample(min(len(g), cap), random_state=SEED) for _, g in S.groupby("class_id")])
    out = f"{ROOT}/data/real_splits/cache/{part}"; os.makedirs(out, exist_ok=True); rows = []
    for shard, g in sel.groupby("shard"):
        pf = pq.ParquetFile(f"{SAGE}/{shard}")
        for rg, h in g.groupby("row_group"):
            col = pf.read_row_group(int(rg), columns=["image"]).column("image")
            for _, r in h.iterrows():
                p = f"{out}/{r.sha1}.jpg"
                if not os.path.exists(p):
                    im = Image.open(io.BytesIO(col[int(r.row_in_group)].as_py()["bytes"])).convert("RGB"); im.thumbnail((512, 512)); im.save(p, quality=92)
                rows.append(dict(path=os.path.relpath(p, ROOT), sha1=r.sha1, class_id=r.class_id, source=r.source, group=r.group))
    M = pd.DataFrame(rows); M["cap"] = cap; M["seed"] = SEED; M["partition"] = part
    M.to_csv(f"{ROOT}/data/real_splits/{name}.csv", index=False)
    print(len(classes), "classes;", len(M), "images; per class min/median/max",
          M.groupby("class_id").size().min(), int(M.groupby("class_id").size().median()), M.groupby("class_id").size().max())


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4])
