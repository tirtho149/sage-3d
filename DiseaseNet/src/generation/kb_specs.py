"""Stage 0 step 3: KB edit specifications for every Stage-0 class (reuses SAGE-Extend's
kb_grounding.structured: verbatim-quote phrases, one disease stage, hedged phrases excluded,
geometry operators only where implemented).

usage: ~/Desktop/Research/.venv/bin/python -m src.generation.kb_specs  -> data/synthetic/stage0/kb_specs.json
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.expanduser("~/Desktop/Research/external/SAGE-Extend")
sys.path.insert(0, f"{EXT}/src")
# organ shown in the chosen parent photos (contact sheets in data/synthetic/stage0/); default leaf
ORGAN = {"corn::smut": "ear", "citrus::canker": "fruit"}


# DiseaseNet override (documented): "may be observed / visible / seen" hedges VISIBILITY, not the symptom.
# SAGE-Extend's HEDGE rule drops such sentences (e.g. grape black rot: "Round, tan lesions ... may be observed
# on leaves"), which left only fragments. Every other hedge (may/might/can/sometimes/...) still excludes a phrase.
HEDGE_DN = (r"\b(may(?! be (?:observed|visible|seen))|might|can(?! be (?:observed|seen))|sometimes|occasionally|"
            r"often|in severe|severe(ly)? infect\w*|less commonly|if)\b")


def main():
    import kb_grounding.structured as structured
    structured.HEDGE = HEDGE_DN
    from kb_grounding.structured import build
    classes = [c["class_id"] for c in json.load(open(f"{ROOT}/configs/datasets/stage0_classes.json"))["classes"]]
    M = pd.read_parquet(f"{ROOT}/data/manifests/sage_images.parquet", columns=["class_id", "crop", "disease", "has_kb"])
    out = {}
    for cid in classes:
        g = M[(M.class_id == cid) & M.has_kb]
        crop, dis = g.groupby(["crop", "disease"]).size().idxmax()           # raw names used by the KB table
        organ = ORGAN.get(cid, "leaf")
        try:
            s = build(crop, dis, organ)
        except Exception as e:  # noqa: BLE001
            out[cid] = dict(error=repr(e)[:300], crop=crop, disease=dis, organ=organ); print(cid, "ERROR", e); continue
        keep = {k: s[k] for k in ("crop", "condition", "target_organ", "stage", "n_stages", "source_url", "geometry", "appearance", "prohibited", "uncertain", "prompts")}
        out[cid] = keep
        ops = [o["op"] for o in s["geometry"]["operators"]]; un = [o["op"] for o in s["geometry"]["unsupported_operators"]]
        print(f"{cid:38s} organ={organ:7s} stage={s['stage']}/{s['n_stages']} ops={ops} unsupported={un}\n   kb: {(s['prompts']['kb_joint_stage']['text'] or '')[:150]}")
    json.dump(out, open(f"{ROOT}/data/synthetic/stage0/kb_specs.json", "w"), indent=1)


if __name__ == "__main__":
    main()
