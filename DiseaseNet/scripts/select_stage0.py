"""Phase 0 step 6: choose the 20 Stage-0 classes BEFORE any result exists (protocol Part M rules).

Rules (declared here; nothing below depends on classifier results):
  * pool = known (not open-set) tier-A classes (B or C, >= 100 unique images), minus
    pest/non-pathogen labels and the dataset-artefact label "alternaria_black_molds_stem_cankers";
  * within-crop pairs, so the task cannot be solved by recognising the crop alone:
      - the 5 crops with existing SAM 3D assets keep their pilot disease plus the most frequent
        other pool class of that crop with the OPPOSITE modality if one exists (else same);
      - 5 further crops with >= 2 pool classes, drawn with a fixed seed; per crop one geometry (C)
        and one appearance (B) class when both exist, drawn with the same seed;
  * nested later stages must contain these 20 (protocol Part M).
usage: ~/Desktop/Research/.venv/bin/python scripts/select_stage0.py -> configs/datasets/stage0_classes.json
"""
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 20261015
PILOTS = {"peach": "peach::leaf_curl", "tomato": "tomato::leaf_mold", "corn": "corn::smut",
          "citrus": "citrus::canker", "mango": "mango::anthracnose"}
PEST = r"mite|attack|insect|aphid|caterpillar|beetle|borer|worm|thrip|weevil|hopper|scale_insect|nematode"
ARTEFACT = {"alternaria_black_molds_stem_cankers"}


def main():
    C = pd.read_parquet(f"{ROOT}/metadata/class_ontology.parquet")
    L = json.load(open(f"{ROOT}/data/real_splits/LOCK.json"))
    pool = C[C.tier.isin(["B", "C"]) & C.tier_A & ~C.class_id.isin(L["open_set_classes"])].copy()
    pool["crop_id"] = pool.class_id.str.split("::").str[0]; pool["dis_id"] = pool.class_id.str.split("::").str[1]
    pool = pool[~pool.dis_id.str.contains(PEST) & ~pool.dis_id.isin(ARTEFACT)]
    chosen, why = [], {}
    for crop, pid in PILOTS.items():
        assert pid in set(pool.class_id), pid
        chosen.append(pid); why[pid] = "pilot class with existing SAM 3D assets"
        mod = pool.set_index("class_id").tier[pid]
        rest = pool[(pool.crop_id == crop) & (pool.class_id != pid)].sort_values("real_image_count", ascending=False)
        opp = rest[rest.tier != mod]
        pick = (opp if len(opp) else rest).iloc[0].class_id
        chosen.append(pick); why[pick] = f"second {crop} disease ({'opposite' if len(opp) else 'same'} modality, most frequent)"
    rng = np.random.default_rng(SEED)
    others = pool[~pool.crop_id.isin(PILOTS)]
    crops = sorted(c for c, g in others.groupby("crop_id") if len(g) >= 2)
    for crop in rng.choice(crops, 5, replace=False):
        g = others[others.crop_id == crop]
        cc, bb = g[g.tier == "C"], g[g.tier == "B"]
        if len(cc) and len(bb):
            picks = [cc.iloc[rng.integers(len(cc))].class_id, bb.iloc[rng.integers(len(bb))].class_id]
        else:
            picks = list(g.iloc[rng.choice(len(g), 2, replace=False)].class_id)
        for p in picks:
            chosen.append(p); why[p] = f"seeded crop draw ({crop})"
    S = pool.set_index("class_id").loc[chosen]
    out = dict(seed=SEED, rules=__doc__.split("Rules")[1].split("usage")[0].strip(), classes=[
        dict(class_id=c, tier=S.tier[c], modality="geometry" if S.tier[c] == "C" else "appearance",
             unique_images=int(S.real_image_count[c]), geometry_terms=S.geometry_terms[c], reason=why[c]) for c in chosen],
        n_crops=int(S.reset_index().class_id.str.split("::").str[0].nunique()),
        n_geometry=int((S.tier == "C").sum()), n_appearance=int((S.tier == "B").sum()))
    os.makedirs(f"{ROOT}/configs/datasets", exist_ok=True)
    json.dump(out, open(f"{ROOT}/configs/datasets/stage0_classes.json", "w"), indent=1)
    for c in out["classes"]:
        print(f"{c['class_id']:42s} {c['modality']:10s} {c['unique_images']:6d}  {c['reason']}")
    print(f"crops={out['n_crops']} geometry={out['n_geometry']} appearance={out['n_appearance']}")


if __name__ == "__main__":
    main()
