"""Exact-key retrieval of SAGE KB records from the verified shard manifest.

Retrieval is by (crop, disease) label, not by semantic search: in this study the
disease label of each input image is known, and the question is what the KB
adds *given* the label. This removes retrieval error as a confound.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import pandas as pd
from .schema import KBRecord
from .extract import extract_specs

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "metadata" / "sage_shard_manifest.parquet"


def _clean(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    x = str(x).strip()
    return None if x == "" or x.lower() == "nan" else x


@lru_cache(maxsize=1)
def _table(path: str = str(MANIFEST)) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["crop", "disease", "canonical_disease", "plant_organ", "visual_symptoms",
                                        "pathogen", "disease_type", "symptom_source", "symptom_quote", "filename"])
    df = df.drop_duplicates(["crop", "disease", "filename"])
    n = df.groupby(["crop", "disease"]).size().rename("n_images")
    kb = df[df.visual_symptoms.map(_clean).notna()].groupby(["crop", "disease"]).first().join(n)
    return kb


def get_record(crop: str, disease: str, with_specs: bool = True) -> KBRecord | None:
    kb = _table()
    if (crop, disease) not in kb.index:
        return None
    r = kb.loc[(crop, disease)]
    organs = [o.strip(" '\"[]").lower() for o in (_clean(r.plant_organ) or "").split(",")]
    organs = [o for o in organs if o]
    rec = KBRecord(crop=crop, disease=disease, canonical_disease=_clean(r.canonical_disease), organs=organs,
                   pathogen=_clean(r.pathogen), disease_type=_clean(r.disease_type),
                   visual_symptoms=_clean(r.visual_symptoms) or "", symptom_quote=_clean(r.symptom_quote) or "",
                   source_url=_clean(r.symptom_source) or "", n_images=int(r.n_images))
    if with_specs:
        rec.specs = extract_specs(rec.visual_symptoms, rec.symptom_quote, rec.source_url)
    return rec


def all_keys():
    return list(_table().index)
