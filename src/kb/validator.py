"""Validation layer: no edit may be applied unless it is backed by evidence.

Checks
  V1  every spec has provenance whose span occurs verbatim in the cited field
  V2  'observed' status requires an explicit per-specimen annotation
  V3  'unsupported' specs are never passed to an editor
  V4  every edit operation in a plan references a spec that passed V1–V3
  V5  edit target organ must be one the KB lists for the disease (if listed)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from .schema import KBRecord, SymptomSpec, Evidence

ORGAN_ALIASES = {"leaf": {"foliar", "leaf", "leaves"}, "stem": {"stem", "vascular", "stolon"},
                 "fruit": {"fruit", "pod", "seed", "ear", "head", "tuber"}, "root": {"root"},
                 "whole_plant": {"whole plant"}}


@dataclass
class ValidationReport:
    passed: list[SymptomSpec] = field(default_factory=list)
    rejected: list[tuple[SymptomSpec, str]] = field(default_factory=list)

    def summary(self):
        return {"passed": len(self.passed), "rejected": len(self.rejected),
                "reasons": sorted({r for _, r in self.rejected})}


def _organ_allowed(rec: KBRecord, organ: str) -> bool:
    if not rec.organs or organ in ("unknown",):
        return True
    return bool(ORGAN_ALIASES.get(organ, {organ}) & set(rec.organs))


def validate(rec: KBRecord, observed_annotations: dict[int, bool] | None = None) -> ValidationReport:
    """observed_annotations: spec index -> True if an annotator confirmed it in this specimen."""
    rep = ValidationReport()
    observed_annotations = observed_annotations or {}
    texts = {"symptom_quote": rec.symptom_quote, "visual_symptoms": rec.visual_symptoms}
    for i, s in enumerate(rec.specs):
        p = s.provenance
        if p is None or not p.span or p.span not in texts.get(p.field_name, ""):
            s.evidence = Evidence.UNSUPPORTED
            rep.rejected.append((s, "V1:no_verbatim_span")); continue
        if observed_annotations.get(i):
            s.evidence = Evidence.OBSERVED
        elif s.evidence == Evidence.OBSERVED:
            rep.rejected.append((s, "V2:observed_without_annotation")); continue
        if not _organ_allowed(rec, s.organ):
            rep.rejected.append((s, "V5:organ_not_in_kb")); continue
        rep.passed.append(s)
    return rep


def check_plan(plan: list[dict], report: ValidationReport) -> list[str]:
    """V4: each plan op must carry spec_index into report.passed. Returns list of errors."""
    errs = []
    for k, op in enumerate(plan):
        j = op.get("spec_index")
        if j is None or not (0 <= j < len(report.passed)):
            errs.append(f"op{k}:{op.get('op')}:unbacked")
        elif report.passed[j].evidence == Evidence.UNSUPPORTED:
            errs.append(f"op{k}:{op.get('op')}:unsupported")
    return errs
