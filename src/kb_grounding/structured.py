"""Stage-aware structured symptom specifications from the SAGE KB (brief v2 section 6).

Only the verbatim web quote (`symptom_quote`) is evidence; `visual_symptoms` (an extraction) is not used for edits.
Rules (deterministic, versioned):
  * Stages: quote sentences in order; a sentence opening with a temporal marker (Later / Shortly after /
    As the disease progresses / As ... age / Eventually / Then) starts a new stage. Edits use ONE stage.
  * Phrases: sentences split at commas, semicolons, parentheses and ", and" so that geometric and appearance
    claims can be separated without paraphrasing. Every prompt is template + verbatim phrases.
  * Geometry operators (src/geometry_editor/deform.py) from lexicon terms; terms with no implemented operator are
    recorded as `unsupported_operator` (never silently dropped, never approximated).
  * Hedged phrases (may/sometimes/...) get evidence `uncertain` and are excluded from edits by default.
  * Features of other stages are listed as prohibited for the chosen stage.
usage: python -m kb_grounding.structured  -> data/kb/structured_specs.json
"""
from __future__ import annotations
import re, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kb.retriever import get_record
from kb.extract import HEDGE, ORGANS, SURFACE

TEMPORAL = re.compile(r"^(later|shortly after|as the disease progresses|as (the )?(leaves|lesions|fruit|plants?) (age|mature)|eventually|then|subsequently|as they mature)\b", re.I)
GEOM_OPS = [  # (regex, operator, implemented)
    (r"\bpucker\w*|\bblister\w*", "pucker", True), (r"\bdistort\w*|\bdeform\w*|\bmalform\w*", "distort", True),
    (r"\bthicken\w*", "thicken", True), (r"\bgall\w*|\btumou?r\w*|\bswell\w*|\bswollen|\bknot\w*", "outgrowth", True),
    (r"\bcrater\w*|\bdepress\w*|\bsunken", "depress", True), (r"\braised\b|\bwarty\b", "raise", False),
    (r"\bcurl\w*|\broll\w*|\bcupp\w*", "curl", False), (r"\bwither\w*|\bwilt\w*", "wilt", False), (r"\bcork\w*", "corky_texture", False),
]
APPEAR = re.compile(r"\b(yellow\w*|chloro\w*|necro\w*|brown\w*|black\w*|purpl\w*|red\w*|bronz\w*|mottl\w*|pale|white|gr[ae]y|olive\w*|oily|shiny|sooty|powdery|velvety|mold|mould|spots?|lesions?|halo|cast|mildew)\b", re.I)
DROP = re.compile(r"\b(drop|die|dies|remain attached)\b", re.I)


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t or "") if len(s.strip()) > 3]


def phrases(sent):
    parts = re.split(r"\s*(?:\(|\)|;|,\s*and\s+|,|\s+and\s+(?=(?:may|might|can|sometimes|often)\b))\s*", sent.rstrip("."))
    return [p.strip() for p in parts if len(p.strip()) > 2]


def organs_in(text):
    return [o for o, pat in ORGANS.items() if o != "whole_plant" and re.search(pat, text, re.I)]


VERB = re.compile(r"\b(have|has|are|is|become|becomes|form|forms|develop|develops|turn|turns|appear|appears|show|shows)\b", re.I)


def build(crop, disease, organ):
    r = get_record(crop, disease); q = r.symptom_quote; stage = 0; out = []
    for s in sentences(q):
        if out and TEMPORAL.search(s): stage += 1
        sorg = organs_in(s); ph = phrases(s)
        m = VERB.search(ph[0]) if ph else None; subj = ph[0][:m.start()].strip() if m and m.start() > 0 else None
        for p in ph:
            assert p in q, (p, q)                                  # verbatim guarantee
            geo = [(op, impl, m.group(0)) for pat, op, impl in GEOM_OPS for m in [re.search(pat, p, re.I)] if m]
            app = [m.group(0) for m in APPEAR.finditer(p)]
            porg = organs_in(p)
            out.append(dict(stage=stage, sentence=s, phrase=p, subject=subj, organs=porg or (sorg if len(sorg) == 1 else []), surface=next((k for k, pt in SURFACE if re.search(pt, s, re.I)), "unspecified"),
                            geometry=[dict(op=o, implemented=i, term=t) for o, i, t in geo], appearance=app,
                            hedged=bool(re.search(HEDGE, p, re.I)), terminal=bool(DROP.search(p))))
    # choose stage: earliest stage with any non-hedged claim on the target organ (or organ-unspecified)
    rel = lambda x: (not x["organs"] or organ in x["organs"] or (organ == "ear" and "fruit" in x["organs"])) and not x["hedged"] and not x["terminal"] and (x["geometry"] or x["appearance"])
    stages = sorted({x["stage"] for x in out if rel(x)}); st = stages[0] if stages else None
    cur = [x for x in out if x["stage"] == st and rel(x)]
    geo = [dict(g, phrase=x["phrase"]) for x in cur for g in x["geometry"]]
    spec = dict(crop=crop, condition=disease, target_organ=organ, stage=st, n_stages=(max(x["stage"] for x in out) + 1) if out else 0,
                source_url=r.source_url, source_quote=q, evidence_status="source_supported_unobserved",
                geometry=dict(operators=[g for g in geo if g["implemented"]], unsupported_operators=[g for g in geo if not g["implemented"]]),
                appearance=[dict(terms=x["appearance"], phrase=x["phrase"], surface=x["surface"]) for x in cur if x["appearance"] and not x["geometry"]],
                prohibited=[dict(stage=x["stage"], phrase=x["phrase"]) for x in out if x["stage"] != st and (x["geometry"] or x["appearance"])]
                           + [dict(rule="no geometry operator without a cited phrase"), dict(rule="preserve background and unaffected organs")],
                uncertain=[x["phrase"] for x in out if x["hedged"]], phrases=out)
    pre = f"make this {crop.lower()} {organ} show {disease.lower()}"
    verbs = re.compile(r"^(and\s+)?(have|has|are|is|become|form|develop|turn)\b", re.I)
    plural = {"leaf": "leaves", "fruit": "fruits", "ear": "ears"}.get(organ, organ + "s")
    subj_of = {x["phrase"]: x["subject"] for x in out}
    fix = lambda p: (f"{subj_of.get(p) or plural} " if verbs.search(p) else "") + re.sub(r"^and\s+", "", p)
    jo = [x["phrase"] for x in cur]; ap = [x["phrase"] for x in cur if x["appearance"] and not x["geometry"]]
    go = [x["phrase"] for x in cur if x["geometry"] and not x["appearance"]]
    def mk(ph):
        return None if not ph else f"{pre}: " + "; ".join(fix(p)[0].lower() + fix(p)[1:] for p in ph)
    spec["prompts"] = {
        "generic": dict(text=f"add disease symptoms to the {organ}", spans=[]),
        "kb_joint_stage": dict(text=mk(jo), spans=jo),
        "kb_appearance_only": dict(text=mk(ap), spans=ap, na_reason=None if ap else "no appearance-only phrase in the selected stage"),
        "kb_geometry_only": dict(text=mk(go), spans=go, na_reason=None if go else "no geometry-only phrase in the selected stage"),
    }
    return spec


PILOT = [("Peach", "Leaf Curl", "leaf"), ("Tomato", "Leaf Mold", "leaf"), ("Corn", "Smut", "ear"), ("Citrus", "Canker", "fruit"), ("Citrus", "Canker", "leaf"), ("Mango", "Anthracnose", "leaf")]

if __name__ == "__main__":
    S = {f"{c}__{d.replace(' ', '_')}" + ("" if (c, o) != ("Citrus", "leaf") else "__leaf"): build(c, d, o) for c, d, o in PILOT}
    os.makedirs("data/kb", exist_ok=True); json.dump(S, open("data/kb/structured_specs.json", "w"), indent=1, ensure_ascii=False)
    for k, s in S.items():
        print(f"== {k} stage {s['stage']}/{s['n_stages']}  ops={[g['op'] for g in s['geometry']['operators']]} unsupported={[g['op'] for g in s['geometry']['unsupported_operators']]}")
        for c, p in s["prompts"].items(): print(f"   {c}: {p['text']}")
