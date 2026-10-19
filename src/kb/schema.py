"""Structured, provenance-tracked symptom specifications built from the SAGE KB.

A SAGE KB row gives free text (`visual_symptoms`), a verbatim `symptom_quote`
and a `symptom_source` URL per (crop, disease). We turn that into a list of
`SymptomSpec`s, each of which must cite a span of the verbatim quote or of the
visual_symptoms text (see validator.py).

Evidence status is about a *specific specimen*, not the disease in general:
  observed                    – confirmed visible in this specimen's input image
                                (only set by an annotator/judge, never by the KB)
  source_supported_unobserved – the source says the disease can show it; not
                                confirmed for this specimen
  uncertain                   – source hedges ("may", "sometimes", "in severe cases")
  unsupported                 – no supporting span; must never be edited in
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum


class Evidence(str, Enum):
    OBSERVED = "observed"
    SOURCE_SUPPORTED_UNOBSERVED = "source_supported_unobserved"
    UNCERTAIN = "uncertain"
    UNSUPPORTED = "unsupported"


class FeatureType(str, Enum):
    COLOR = "color"                # chlorosis, necrosis, discoloration
    LESION = "lesion"              # spots / blotches with a boundary
    SURFACE_GROWTH = "surface_growth"  # mildew, mold, sporulation, pustules
    DEFORMATION = "deformation"    # curl, roll, cup, pucker, twist
    OUTGROWTH = "outgrowth"        # gall, knot, swelling, tumor
    RELIEF = "relief"              # sunken / raised / cracked / corky / crater
    ARCHITECTURE = "architecture"  # witches' broom, rosetting, proliferation, stunting


# Feature types whose effect is primarily geometric (needs 3D to represent).
GEOMETRIC = {FeatureType.DEFORMATION, FeatureType.OUTGROWTH, FeatureType.RELIEF, FeatureType.ARCHITECTURE}


@dataclass
class Provenance:
    source_url: str
    span: str                 # exact substring of `field_name` text
    field_name: str           # "symptom_quote" (verbatim web quote) or "visual_symptoms"
    verbatim: bool            # True iff span is found in symptom_quote


@dataclass
class SymptomSpec:
    feature: FeatureType
    organ: str                       # leaf | stem | fruit | root | whole_plant | unknown
    surface: str = "unspecified"     # upper | lower | both | unspecified
    attributes: dict = field(default_factory=dict)  # e.g. {"direction": "upward", "color": "brown"}
    evidence: Evidence = Evidence.SOURCE_SUPPORTED_UNOBSERVED
    hedged: bool = False
    provenance: Provenance | None = None

    @property
    def geometric(self) -> bool:
        return self.feature in GEOMETRIC

    def to_dict(self):
        d = asdict(self)
        d["feature"] = self.feature.value
        d["evidence"] = self.evidence.value
        d["geometric"] = self.geometric
        return d


@dataclass
class KBRecord:
    crop: str
    disease: str
    canonical_disease: str | None
    organs: list[str]
    pathogen: str | None
    disease_type: str | None
    visual_symptoms: str
    symptom_quote: str
    source_url: str
    n_images: int = 0
    specs: list[SymptomSpec] = field(default_factory=list)

    @property
    def key(self):
        return (self.crop, self.disease)

    def to_dict(self):
        d = asdict(self)
        d["specs"] = [s.to_dict() for s in self.specs]
        return d
