"""Edit instructions for every arm, built from the SAME information.

  generic        – disease-agnostic instruction (arms B, E)
  kb             – instruction built only from validated specs (arms C, F)
  kb_no_geometry – kb minus geometric specs (ablation)
  kb_mismatched  – kb of a different disease of the same crop (negative control)

2D and 3D arms receive identical text for the same condition, so any
difference between them is attributable to the representation, not the prompt.
"""
from __future__ import annotations
import re
from .schema import KBRecord, FeatureType, GEOMETRIC
from .extract import LEXICON
from .validator import ValidationReport

GENERIC = "add disease symptoms to the {organ}"
MAX_CLAUSES = 2   # CLIP text encoder truncates at 77 tokens; fixed for all arms (see token check in tests)


MAX_WORDS = 22


def _phrase(s) -> str:
    """The cited source clause itself, trimmed around the matched term (no paraphrase)."""
    words = s.provenance.span.rstrip(".;").split()
    terms = s.attributes.get("terms", [])
    hit = next((i for i, w in enumerate(words) if any(w.lower().strip(",()").startswith(t[:5]) for t in terms)), 0)
    lo = max(0, min(hit - MAX_WORDS // 2, len(words) - MAX_WORDS))
    return " ".join(words[lo:lo + MAX_WORDS]).strip(",; ").lower()


def _order(specs):
    # geometric first, then surface growth, lesion, color; non-hedged before hedged
    rank = {FeatureType.DEFORMATION: 0, FeatureType.OUTGROWTH: 0, FeatureType.RELIEF: 1, FeatureType.ARCHITECTURE: 1,
            FeatureType.SURFACE_GROWTH: 2, FeatureType.LESION: 3, FeatureType.COLOR: 4}
    return sorted(specs, key=lambda s: (s.hedged, rank[s.feature]))


ORGAN_NAME = {("Corn", "fruit"): "ear"}


def _words(t):
    return set(re.findall(r"[a-z]+", t.lower()))


def _jaccard(a, b):
    A, B = _words(a), _words(b)
    return len(A & B) / max(1, len(A | B))


def build(condition: str, rec: KBRecord, report: ValidationReport, organ: str = "leaf",
          mismatched: tuple[KBRecord, ValidationReport] | None = None) -> dict:
    oname = ORGAN_NAME.get((rec.crop, organ), organ)
    if condition == "generic":
        return {"condition": condition, "text": GENERIC.format(organ=oname), "spec_ids": []}
    if condition == "kb_mismatched":
        assert mismatched is not None, "need a mismatched record"
        rec, report = mismatched
    specs = [s for s in report.passed if s.organ in (organ, "unknown")]
    organ_fallback = False
    if condition == "kb_mismatched" and not specs:
        specs, organ_fallback = list(report.passed), True   # wrong-disease knowledge, any organ
    if condition == "kb_no_geometry":
        # drop every clause that carries a geometric claim, even if it also carries colour
        geo = "|".join(LEXICON[f] for f in GEOMETRIC)
        specs = [s for s in specs if not s.geometric and not re.search(geo, s.provenance.span, re.I)]
    firm = [s for s in specs if not s.hedged]
    specs = _order(firm if firm else specs)   # hedged claims only if nothing firmer exists
    clauses = []
    for s in specs:
        p = _phrase(s)
        if p and all(_jaccard(p, q) < 0.5 for q in clauses):   # drop near-repeats (quote vs paraphrase)
            clauses.append(p)
    clauses = clauses[:MAX_CLAUSES]
    if not clauses:
        return {"condition": condition, "text": None, "spec_ids": [], "na_reason": "no validated clause for this organ/condition"}
    text = f"make this {rec.crop.lower()} {oname} show {rec.disease.lower()}: " + "; ".join(clauses)
    return {"condition": condition, "text": text, "spec_ids": [report.passed.index(s) for s in specs],
            "source_url": rec.source_url, "disease": rec.disease, "organ_fallback": organ_fallback}


def pick_mismatched(rec: KBRecord, organ: str, candidates: list[tuple[KBRecord, ValidationReport]]):
    """Most frequent other disease of the same crop whose KB yields a non-empty prompt for `organ`."""
    for mr, mv in sorted(candidates, key=lambda x: -x[0].n_images):
        if mr.disease == rec.disease:
            continue
        # a valid negative control must carry DIFFERENT knowledge (SAGE has some copied KB texts)
        if _jaccard(mr.visual_symptoms + " " + mr.symptom_quote, rec.visual_symptoms + " " + rec.symptom_quote) > 0.3:
            continue
        if build("kb", mr, mv, organ=organ)["text"]:
            return mr, mv
    for mr, mv in sorted(candidates, key=lambda x: -x[0].n_images):   # fallback: any organ
        if mr.disease != rec.disease and mv.passed and _jaccard(mr.visual_symptoms + " " + mr.symptom_quote, rec.visual_symptoms + " " + rec.symptom_quote) <= 0.3:
            return mr, mv
    return None
