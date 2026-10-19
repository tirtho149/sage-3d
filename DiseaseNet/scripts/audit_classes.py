"""Phase 0 / Part C: audit SAGE class coverage from the parquet shards (not the stale parquet).

Reads every shard (image bytes are hashed, not decoded), builds
  data/manifests/sage_images.parquet   one row per image: class_id, sha1, source, shard, row, labels
  metadata/class_ontology.parquet       one row per crop::disease class (protocol C1 fields + tier)
  metadata/class_coverage.csv           same, CSV
and prints the headline counts used by reports/02_sage_dataset_audit.md.

Eligibility (protocol C2), decided from the verbatim KB text and the operators that exist
in SAGE-Extend today (src/geometry_editor: deform.py KB terms, curl.py, gall.py, bilayer.py):
  geometry_terms  -> modality "geometry" (Tier C) when the organ is reconstructable
  appearance only -> modality "appearance" (Tier B)
  Tier D          -> no KB text / no canonical crop / healthy / organ not reconstructable /
                     < MIN_UNIQUE unique images / label conflict share too high
  Tier A flag     -> B or C with >= MIN_A unique images (enough for locked test + parent assets)
The keyword map is a transparent first pass, recorded in the report; it is not a biological
judgement and must be checked (protocol F2).

usage: ~/Desktop/Research/.venv/bin/python scripts/audit_classes.py
"""
import glob
import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAGE = os.path.expanduser("~/Desktop/Research/data/raw/SAGE/data")
META_COLS = ["crop", "disease", "canonical_disease", "plant_organ", "visual_symptoms", "pathogen",
             "disease_type", "symptom_source", "symptom_quote", "source", "filename", "raw_label"]
MIN_UNIQUE = 30          # below this a class cannot supply both parent assets and a locked test set
MIN_A = 100

GEOMETRY_TERMS = {
    "pucker": r"pucker|blister|crinkl",
    "distort": r"distort|deform|malform|twist",
    "thicken": r"thicken",
    "gall": r"\bgall|tumou?r|swelling|swollen|outgrowth|hypertroph",
    "curl": r"\bcurl|\broll(ed|ing)?\b|cupp?(ed|ing)",
    "sunken": r"sunken|crater|depress",
}
APPEARANCE_TERMS = r"spot|lesion|blotch|mottl|mosaic|chloro|yellow|necro|brown|black|mold|mould|mildew|rust|blight|streak|ring|halo|discolo|powder|sooty|scab|canker|wilt|rot|pustule|stripe|speck"
RECONSTRUCTABLE = r"leaf|leaves|foliar|foliage|fruit|ear|cob|pod|tuber|head|berry|needle|frond|blade"
NOT_RECONSTRUCTABLE = r"^(root|roots|vascular|whole plant|soil|seed)$"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def read_shard(path):
    pf = pq.ParquetFile(path); out = []
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=META_COLS + ["image"]).to_pandas()
        sha = [hashlib.sha1(im["bytes"]).hexdigest() if isinstance(im, dict) and im.get("bytes") else None
               for im in t["image"]]
        t = t.drop(columns=["image"]); t["sha1"] = sha
        t["shard"] = os.path.basename(path); t["row_group"] = rg; t["row_in_group"] = np.arange(len(t))
        out.append(t)
    return pd.concat(out, ignore_index=True)


def main():
    os.makedirs(f"{ROOT}/data/manifests", exist_ok=True); os.makedirs(f"{ROOT}/metadata", exist_ok=True)
    shards = sorted(glob.glob(f"{SAGE}/train-*.parquet"))
    with ProcessPoolExecutor(6) as ex:
        df = pd.concat(list(ex.map(read_shard, shards)), ignore_index=True)
    df["crop_id"] = df.crop.map(slug); df["disease_id"] = df.canonical_disease.fillna(df.disease).map(slug)
    df["class_id"] = df.crop_id + "::" + df.disease_id
    df["has_kb"] = df.visual_symptoms.fillna("").str.len().gt(0) & df.symptom_source.fillna("").str.len().gt(0)
    df.to_parquet(f"{ROOT}/data/manifests/sage_images.parquet", index=False)

    n_rows = len(df); n_img = df.sha1.notna().sum()
    dup = df[df.sha1.notna()].groupby("sha1")
    n_unique = dup.ngroups
    multi_label = dup.class_id.nunique(); conflict_sha = set(multi_label[multi_label > 1].index)

    rows = []
    for cid, g in df.groupby("class_id"):
        gi = g[g.sha1.notna()]
        uniq = gi.sha1.nunique()
        conflicts = gi.sha1.isin(conflict_sha).mean() if len(gi) else 0.0
        kb = g[g.has_kb]
        text = " ".join(kb.visual_symptoms.dropna().unique()[:5]).lower()
        # plant_organ comes as "Leaf, Stem" or as stringified lists "['foliar', 'stem']": strip list syntax
        organs = sorted({o.strip().strip("'\"").lower() for s in g.plant_organ.dropna()
                         for o in re.sub(r"[\[\]]", ",", str(s)).split(",") if o.strip().strip("'\"")})
        geo = [k for k, pat in GEOMETRY_TERMS.items() if re.search(pat, text)]
        app = bool(re.search(APPEARANCE_TERMS, text))
        recon = any(re.search(RECONSTRUCTABLE, o) for o in organs)
        only_bad = organs and all(re.search(NOT_RECONSTRUCTABLE, o) for o in organs)
        healthy = "healthy" in cid
        crop_ok = bool(g.crop_id.iloc[0]) and g.crop_id.iloc[0] not in ("none", "nan")
        reason = ""
        if not crop_ok: reason = "no canonical crop"
        elif healthy: reason = "healthy class (control, not a disease)"
        elif len(kb) == 0: reason = "no source-grounded KB text"
        elif only_bad or not recon: reason = f"organ not reconstructable ({', '.join(organs) or 'unknown'})"
        elif uniq < MIN_UNIQUE: reason = f"< {MIN_UNIQUE} unique images"
        elif conflicts > 0.2: reason = "> 20% of images carry conflicting labels"
        elif not geo and not app: reason = "no supported symptom term"
        if reason: tier, method = "D", "none"
        elif geo: tier, method = "C", "3D geometry (" + "+".join(geo) + ") + appearance"
        else: tier, method = "B", "appearance-only"
        rows.append(dict(class_id=cid, crop_name=g.crop.iloc[0], disease_name=g.canonical_disease.fillna(g.disease).iloc[0],
                         pathogen_type=g.disease_type.dropna().iloc[0] if g.disease_type.notna().any() else None,
                         affected_organs=", ".join(organs), kb_source=kb.symptom_source.dropna().iloc[0] if len(kb) else None,
                         symptom_stage=None, source_dataset=", ".join(sorted(g.source.dropna().unique())[:5]),
                         n_sources=int(g.source.nunique()),
                         real_image_rows=int(len(g)), real_image_count=int(uniq), conflict_frac=float(conflicts),
                         kb_rows=int(len(kb)), geometry_terms="+".join(geo), appearance_terms=bool(app),
                         synthetic_generation_eligible=tier in ("B", "C"), tier=tier,
                         tier_A=bool(tier in ("B", "C") and uniq >= MIN_A),
                         generation_method=method, exclusion_reason=reason))
    C = pd.DataFrame(rows).sort_values(["tier", "real_image_count"], ascending=[True, False])
    C.to_parquet(f"{ROOT}/metadata/class_ontology.parquet", index=False)
    C.to_csv(f"{ROOT}/metadata/class_coverage.csv", index=False)
    summary = dict(rows=int(n_rows), image_rows=int(n_img), unique_images=int(n_unique),
                   exact_duplicate_rows=int(n_img - n_unique),
                   images_with_conflicting_labels=int(len(conflict_sha)),
                   crops=int(df.crop_id.nunique()), classes=int(df.class_id.nunique()),
                   classes_with_kb=int(C.kb_rows.gt(0).sum()),
                   tiers=C.tier.value_counts().to_dict(), tier_A=int(C.tier_A.sum()),
                   tier_A_geometry=int((C.tier_A & (C.tier == "C")).sum()),
                   exclusion_reasons=C[C.tier == "D"].exclusion_reason.str.replace(r"\(.*\)", "(...)", regex=True).value_counts().to_dict(),
                   unique_images_in_eligible=int(C[C.tier.isin(["B", "C"])].real_image_count.sum()))
    json.dump(summary, open(f"{ROOT}/metadata/audit_summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
