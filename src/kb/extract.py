"""Rule-based extraction of SymptomSpecs from SAGE KB text.

Deliberately conservative and transparent: every spec is anchored to an exact
clause of the source text. No LLM is used, so nothing can be invented. Recall
is limited to the lexicon below; the lexicon is part of the protocol and
versioned with the code.
"""
from __future__ import annotations
import re
from .schema import SymptomSpec, FeatureType, Provenance, Evidence

LEXICON: dict[FeatureType, str] = {
    FeatureType.DEFORMATION: r"\b(curl\w*|roll\w*|cupp\w*|pucker\w*|twist\w*|distort\w*|malform\w*|deform\w*|crinkl\w*|wrinkl\w*|blister\w*|shepherd'?s crook)",
    FeatureType.OUTGROWTH: r"\b(gall\w*|knot\w*|swell\w*|swollen|tumou?r\w*)",
    FeatureType.RELIEF: r"\b(sunken|raised|crater\w*|cork\w*|crack\w*|depress\w*|canker\w*|scab\w*)",
    FeatureType.ARCHITECTURE: r"\b(witch\w*'? ?broom\w*|broom\w*|rosett\w*|proliferat\w*|bushy|stunt\w*|dwarf\w*|tiller\w*)",
    FeatureType.SURFACE_GROWTH: r"\b(mildew|mold|mould|fuzz\w*|downy|powdery|velvety|sporulat\w*|pustule\w*|spore mass\w*|fungal growth|telial horns?|aecia)",
    FeatureType.LESION: r"\b(spot\w*|lesion\w*|blotch\w*|ring ?spot\w*|halo\w*|streak\w*|stripe\w*|fleck\w*)",
    FeatureType.COLOR: r"\b(yellow\w*|chloro\w*|necro\w*|brown\w*|black\w*|purpl\w*|red\w*|bronz\w*|mottl\w*|mosaic|pale|white|gr[ae]y)",
}
ORGANS = {
    "leaf": r"\b(lea(f|ves)|foliage|foliar|leaflets?|needles?|blades?)\b",
    "stem": r"\b(stems?|stalks?|vines?|canes?|petioles?|branch\w*|twigs?|trunks?|shoots?|bark|hypocotyls?|stolons?|nodes?)\b",
    "fruit": r"\b(fruits?|berr(y|ies)|pods?|ears?|tubers?|cobs?|kernels?|seeds?|heads?|tassels?)\b",
    "root": r"\b(roots?|crowns?|taproots?)\b",
    "whole_plant": r"\b(plants?|vines|trees?|canopy|seedlings?)\b",
}
SURFACE = [("lower", r"\b(under\s?sides?|lower (leaf )?surfaces?|lower sides?|abaxial|undersurface|beneath)\b"),
           ("upper", r"\b(upper (leaf )?surfaces?|upper sides?|adaxial)\b")]
HEDGE = r"\b(may|might|can|sometimes|occasionally|often|in severe|severe(ly)? infect\w*|less commonly|if)\b"
DIRECTION = r"\b(upward|downward|inward|outward|under|upwards|downwards|along the midvein)\b"
COLORWORDS = r"\b(yellow|brown|black|purple|red|reddish|white|gr[ae]y|orange|tan|olive|bronze|pale green|light green|dark green)\b"

_CLAUSE = re.compile(r"[^.;]+[.;]?")


def _clauses(text: str):
    for m in _CLAUSE.finditer(text or ""):
        c = m.group(0).strip()
        if len(c) > 3:
            yield c


def _organ(clause: str) -> str:
    for org, pat in ORGANS.items():
        if re.search(pat, clause, re.I):
            return org
    return "unknown"


def extract_specs(visual_symptoms: str, symptom_quote: str, source_url: str) -> list[SymptomSpec]:
    specs: list[SymptomSpec] = []
    seen = set()
    for field_name, text in (("symptom_quote", symptom_quote), ("visual_symptoms", visual_symptoms)):
        for clause in _clauses(text):
            for ft, pat in LEXICON.items():
                if not re.search(pat, clause, re.I):
                    continue
                organ = _organ(clause)
                surface = next((s for s, p in SURFACE if re.search(p, clause, re.I)), "unspecified")
                attrs = {}
                if (d := re.findall(DIRECTION, clause, re.I)):
                    attrs["direction"] = sorted({x.lower() for x in d})
                if (c := re.findall(COLORWORDS, clause, re.I)):
                    attrs["colors"] = sorted({x.lower() for x in c})
                attrs["terms"] = sorted({m.group(0).lower() for m in re.finditer(pat, clause, re.I)})
                dedup = (ft, organ, surface, tuple(attrs["terms"]))
                if dedup in seen:      # prefer the verbatim-quote occurrence (visited first)
                    continue
                seen.add(dedup)
                hedged = bool(re.search(HEDGE, clause, re.I))
                specs.append(SymptomSpec(
                    feature=ft, organ=organ, surface=surface, attributes=attrs,
                    evidence=Evidence.UNCERTAIN if hedged else Evidence.SOURCE_SUPPORTED_UNOBSERVED,
                    hedged=hedged,
                    provenance=Provenance(source_url=source_url, span=clause, field_name=field_name,
                                          verbatim=clause in (symptom_quote or "")),
                ))
    return specs
