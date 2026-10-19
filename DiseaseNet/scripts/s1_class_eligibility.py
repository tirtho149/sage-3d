"""Stage 1, step 10: canonical class-eligibility manifest (CPU; no model results used).

Joins the Phase-0 class audit (CLASS_COVERAGE.csv) with the locked real partitions (real_splits.parquet) and a
structured KB spec built for every Phase-0-eligible class (SAGE-Extend kb_grounding.structured, same hedge rule as
Stage 0). Per class: generation method, independent sources, real samples per partition, KB evidence, implemented
geometry operators, appearance phrases, QC status, exclusion reason, and Stage-1 eligibility under the rule below.

Stage-1 eligibility rule (declared before any Stage-1 result; class selection never looks at model performance):
  E1 Phase-0 eligible (tier B or C: source-grounded KB, >= 30 unique images, reconstructable organ, <= 20% conflicts)
  E2 known class (not in the open-set reserve)
  E3 KB spec builds and the selected stage's joint KB prompt cites >= 1 non-hedged phrase (usable for a 2D edit)
  E4 >= 12 near-duplicate groups in the `parent` partition (independent parent pool for diversity scaling + IID)
  Organ: Stage-0 overrides (corn smut = ear, citrus canker = fruit), else SAGE plant_organ: "leaf" if Foliar is listed,
     "fruit" if only pod/fruit; the photographed organ is checked again at parent visual QC.
  E5 >= 20 images in the `val` partition (Stage-1 final test) and >= 16 groups in `adapt` (S1 shots + S1 validation)
Outputs: CLASS_ELIGIBILITY.csv, data/synthetic/stage1/kb_specs_all.json
usage: ~/Desktop/Research/.venv/bin/python scripts/s1_class_eligibility.py
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.expanduser("~/Desktop/Research/external/SAGE-Extend")
sys.path.insert(0, f"{EXT}/src"); sys.path.insert(0, ROOT)
from src.generation.kb_specs import HEDGE_DN, ORGAN  # noqa: E402
import kb_grounding.structured as structured  # noqa: E402

structured.HEDGE = HEDGE_DN
MIN_PARENT_GROUPS, MIN_FINAL, MIN_ADAPT_GROUPS = 12, 20, 16


def main():
    C = pd.read_csv(f"{ROOT}/CLASS_COVERAGE.csv")
    R = pd.read_parquet(f"{ROOT}/data/real_splits/real_splits.parquet").drop_duplicates(["sha1", "class_id"])
    lock = json.load(open(f"{ROOT}/data/real_splits/LOCK.json"))
    s0 = {c["class_id"] for c in json.load(open(f"{ROOT}/configs/datasets/stage0_classes.json"))["classes"]}
    img = R.pivot_table(index="class_id", columns="partition", values="sha1", aggfunc="count", fill_value=0)
    grp = R.groupby(["class_id", "partition"]).group.nunique().unstack(fill_value=0)
    M = pd.read_parquet(f"{ROOT}/data/manifests/sage_images.parquet", columns=["class_id", "crop", "disease", "has_kb", "plant_organ"])
    po = M.groupby("class_id").plant_organ.agg(lambda s: str(s.value_counts().index[0]) if s.notna().any() else "")
    specs, rows = {}, []
    for _, c in C.iterrows():
        cid = c.class_id; r = dict(class_id=cid, crop=c.crop_name, disease=c.disease_name, pathogen_type=c.pathogen_type,
                                   phase0_tier=c.tier, phase0_eligible=bool(c.synthetic_generation_eligible),
                                   phase0_exclusion=c.exclusion_reason if isinstance(c.exclusion_reason, str) else "",
                                   kb_source=c.kb_source, n_source_datasets=c.n_sources, conflict_frac=c.conflict_frac,
                                   open_set_reserve=cid in lock["open_set_classes"], stage0_class=cid in s0)
        for p in ("parent", "adapt", "val", "test"):
            r[f"img_{p}"] = int(img.loc[cid, p]) if cid in img.index and p in img.columns else 0
            r[f"groups_{p}"] = int(grp.loc[cid, p]) if cid in grp.index and p in grp.columns else 0
        r.update(kb_spec="", organ="", n_stages=None, stage=None, geometry_ops_implemented="", geometry_ops_unsupported="",
                 n_appearance_phrases=0, n_geometry_phrases=0, n_kb_phrases=0)
        if r["phase0_eligible"]:
            g = M[(M.class_id == cid) & M.has_kb]
            try:
                crop, dis = g.groupby(["crop", "disease"]).size().idxmax()
                organ = ORGAN.get(cid) or ("leaf" if "Foliar" in po.get(cid, "Foliar") else "fruit" if any(
                    k in po.get(cid, "") for k in ("Pod", "Fruit")) else "leaf")            # SAGE plant_organ; verified at parent QC
                s = structured.build(crop, dis, organ)
                ops = [o["op"] for o in s["geometry"]["operators"] if o.get("implemented")]
                un = [o["op"] for o in s["geometry"]["unsupported_operators"]]
                app = s["prompts"]["kb_appearance_only"]["spans"] or []; joint = s["prompts"]["kb_joint_stage"]["spans"] or []
                specs[cid] = {k: s[k] for k in ("crop", "condition", "target_organ", "stage", "n_stages", "source_url", "geometry",
                                                 "appearance", "prohibited", "uncertain", "prompts")}
                r.update(kb_spec="ok", organ=organ, n_stages=s["n_stages"], stage=s["stage"], geometry_ops_implemented="+".join(ops),
                         geometry_ops_unsupported="+".join(un), n_appearance_phrases=len(app), n_geometry_phrases=len(ops), n_kb_phrases=len(joint))
            except Exception as e:  # noqa: BLE001
                r["kb_spec"] = f"error: {repr(e)[:120]}"
        reasons = []
        if not r["phase0_eligible"]:
            reasons.append(f"phase0: {r['phase0_exclusion']}")
        if r["open_set_reserve"]:
            reasons.append("open-set reserve (evaluated only as unknown)")
        if r["phase0_eligible"] and r["kb_spec"] != "ok":
            reasons.append("KB spec failed")
        if r["kb_spec"] == "ok" and r["n_kb_phrases"] == 0:
            reasons.append("no usable KB phrase for the selected stage")
        if r["groups_parent"] < MIN_PARENT_GROUPS:
            reasons.append(f"< {MIN_PARENT_GROUPS} independent parent groups")
        if r["img_val"] < MIN_FINAL:
            reasons.append(f"< {MIN_FINAL} final-test images")
        if r["groups_adapt"] < MIN_ADAPT_GROUPS:
            reasons.append(f"< {MIN_ADAPT_GROUPS} adapt groups")
        r["stage1_eligible"] = not reasons
        r["stage1_exclusion_reason"] = "; ".join(reasons)
        r["synthetic_generation_method"] = ("" if not r["stage1_eligible"] else
                                            ("3D geometry (" + r["geometry_ops_implemented"] + ") + appearance" if r["geometry_ops_implemented"]
                                             else "appearance-only (2D edit; 3D = re-render + lifted edit)"))
        r["symptom_modality"] = ("geometry" if r["geometry_ops_implemented"] else "appearance") if r["kb_spec"] == "ok" else ""
        r["qc_status"] = ("stage0 parents visually QC'd (3/class)" if r["stage0_class"] else
                          "automatic mask QC only at generation time" if r["stage1_eligible"] else "n/a")
        rows.append(r)
    D = pd.DataFrame(rows).sort_values(["stage1_eligible", "groups_parent"], ascending=[False, False])
    D.to_csv(f"{ROOT}/CLASS_ELIGIBILITY.csv", index=False)
    os.makedirs(f"{ROOT}/data/synthetic/stage1", exist_ok=True)
    json.dump(specs, open(f"{ROOT}/data/synthetic/stage1/kb_specs_all.json", "w"), indent=1)
    E = D[D.stage1_eligible]
    print(f"classes {len(D)}; phase0 eligible {int(D.phase0_eligible.sum())}; KB spec ok {int((D.kb_spec == 'ok').sum())}; "
          f"STAGE-1 ELIGIBLE {len(E)} ({E.crop.nunique()} crops); geometry-operator classes {int((E.symptom_modality == 'geometry').sum())}; "
          f"stage0 classes eligible {int(E.stage0_class.sum())}/20")
    print(D[D.phase0_eligible & ~D.stage1_eligible].stage1_exclusion_reason.str.split("; ").explode().value_counts().to_string())
    print(E.groupby("geometry_ops_implemented").size().to_string())


if __name__ == "__main__":
    main()
