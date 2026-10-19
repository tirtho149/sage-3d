"""Amend the Stage-0 class set for KB-validity defects found BEFORE any result (logged, rule-based).

Defects (found while building KB edit specs, src/generation/kb_specs.py):
  rice::bacterial_blight            KB quote describes bacterial PANICLE blight (wrong text), identical to
                                    rice::bacterial_panicle_blight's -> KB cannot distinguish the pair
  rice::bacterial_panicle_blight    shares the identical quote; no usable prompt for the photographed organ
  potato::bacteria_wilt_and_soft_rot KB describes wilting leaves, the class photos show rotting tubers
Rule: replace an invalid class by a valid pool class of the SAME crop (seeded); if the crop has none,
replace the crop pair by a new seeded crop draw (one geometry + one appearance class when possible).
Valid = non-empty leaf prompt from kb_grounding.structured and a quote not shared with another chosen class.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/Desktop/Research/external/SAGE-Extend/src"))
SEED = 20261015 + 1
INVALID = {"rice::bacterial_blight": "KB quote describes bacterial panicle blight; identical to rice::bacterial_panicle_blight",
           "rice::bacterial_panicle_blight": "identical KB quote to rice::bacterial_blight; no prompt for the photographed organ",
           "potato::bacteria_wilt_and_soft_rot": "KB describes leaf wilt; class photos show tuber soft rot (KB-photo mismatch)"}
PEST = r"mite|attack|insect|aphid|caterpillar|beetle|borer|worm|thrip|weevil|hopper|nematode"


def main():
    from kb_grounding.structured import build
    from kb.retriever import get_record
    cfg = json.load(open(f"{ROOT}/configs/datasets/stage0_classes.json"))
    C = pd.read_parquet(f"{ROOT}/metadata/class_ontology.parquet"); L = json.load(open(f"{ROOT}/data/real_splits/LOCK.json"))
    M = pd.read_parquet(f"{ROOT}/data/manifests/sage_images.parquet", columns=["class_id", "crop", "disease", "has_kb"])
    pool = C[C.tier.isin(["B", "C"]) & C.tier_A & ~C.class_id.isin(L["open_set_classes"])].copy()
    pool = pool[~pool.class_id.str.split("::").str[1].str.contains(PEST) & ~pool.class_id.str.contains("alternaria_black_molds")]
    names = lambda cid: M[(M.class_id == cid) & M.has_kb].groupby(["crop", "disease"]).size().idxmax()
    quote = lambda cid: get_record(*names(cid)).symptom_quote
    chosen = [c["class_id"] for c in cfg["classes"] if c["class_id"] not in INVALID]
    used_quotes = {quote(c) for c in chosen}
    valid = lambda cid: cid not in INVALID and cid not in chosen and bool(build(*names(cid), "leaf")["prompts"]["kb_joint_stage"]["text"]) and quote(cid) not in used_quotes
    rng = np.random.default_rng(SEED); amendments = []
    new = []
    for bad in [c for c in INVALID if any(x["class_id"] == c for x in cfg["classes"])]:
        crop = bad.split("::")[0]
        same = [c for c in pool[pool.class_id.str.startswith(crop + "::")].class_id if valid(c)]
        if same:
            pick = same[int(rng.integers(len(same)))]; new.append((bad, pick, "same-crop valid replacement"))
            chosen.append(pick); used_quotes.add(quote(pick))
    gone = [c for c in INVALID if any(x["class_id"] == c for x in cfg["classes"]) and c not in [b for b, _, _ in new]]
    crops_needed = sorted({c.split("::")[0] for c in gone})
    taken = {c.split("::")[0] for c in chosen}
    for crop in crops_needed:
        cands = sorted(c for c, g in pool.groupby(pool.class_id.str.split("::").str[0]) if c not in taken and
                       sum(valid(x) for x in g.class_id) >= 2)
        nc = cands[int(rng.integers(len(cands)))]; taken.add(nc)
        g = pool[pool.class_id.str.startswith(nc + "::")]; g = g[[valid(x) for x in g.class_id]]
        cc, bb = g[g.tier == "C"], g[g.tier == "B"]
        picks = ([cc.iloc[rng.integers(len(cc))].class_id, bb.iloc[rng.integers(len(bb))].class_id] if len(cc) and len(bb)
                 else list(g.iloc[rng.choice(len(g), 2, replace=False)].class_id))
        for b, p in zip([x for x in gone if x.startswith(crop + "::")], picks):
            new.append((b, p, f"crop pair replaced by seeded draw ({nc})")); chosen.append(p); used_quotes.add(quote(p))
    S = pool.set_index("class_id")
    for bad, pick, how in new:
        amendments.append(dict(removed=bad, reason=INVALID[bad], added=pick, rule=how))
    cfg["classes"] = [c for c in cfg["classes"] if c["class_id"] not in INVALID] + [
        dict(class_id=p, tier=S.tier[p], modality="geometry" if S.tier[p] == "C" else "appearance",
             unique_images=int(S.real_image_count[p]), geometry_terms=S.geometry_terms[p], reason=how) for _, p, how in new]
    cfg.setdefault("amendments", []).extend(amendments)
    cfg["n_geometry"] = sum(c["tier"] == "C" for c in cfg["classes"]); cfg["n_appearance"] = sum(c["tier"] == "B" for c in cfg["classes"])
    cfg["n_crops"] = len({c["class_id"].split("::")[0] for c in cfg["classes"]})
    json.dump(cfg, open(f"{ROOT}/configs/datasets/stage0_classes.json", "w"), indent=1)
    for a in amendments: print(a)
    print("classes:", len(cfg["classes"]), "crops:", cfg["n_crops"], "geometry:", cfg["n_geometry"])


if __name__ == "__main__":
    main()
