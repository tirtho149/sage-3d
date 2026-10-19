"""Observed-evidence policy: what to do when the KB text and the photo disagree (review item B2).

Three explicit modes, chosen per edit request:
  preserve        (default) an edit is ACCEPTED only if it does not contradict what the photo
                  shows; otherwise it is REFUSED and nothing is written as a result.
  counterfactual  the edit is allowed to contradict the photo, but every output is LABELLED
                  counterfactual (a simulation of the KB statement, not of this specimen).
  abstain         the KB does not specify the attribute (e.g. no curl direction for Peach):
                  no edit; alternatives may only appear as labelled hypotheses.

"Contradicts the photo" is measured, not judged: the fitted input-view IoU (render at the
corrected input camera vs the REAL organ mask) drops by more than TAU. TAU is the largest
drop produced by appearance-only edits (GaussianEditor E/F, which are not meant to change
shape) on the same specimens: below it, a change is within editing noise. It is computed
from those arms, not from the edits being judged, but it was set after the pilot data
existed, so decisions are also reported at 2*TAU (sensitivity).
"""
from dataclasses import dataclass, asdict
from typing import Optional

MODES = ("preserve", "counterfactual", "abstain")


@dataclass
class Decision:
    decision: str          # accept | refuse | counterfactual | abstain
    mode: str
    iou_before: Optional[float]
    iou_after: Optional[float]
    drop: Optional[float]
    tau: float
    reason: str

    def as_dict(self):
        return asdict(self)


def decide(iou_before, iou_after, *, mode="preserve", tau=0.05, kb_specifies=True):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode == "abstain" or not kb_specifies:
        return Decision("abstain", mode, iou_before, iou_after, None, tau,
                        "KB does not specify this attribute; no edit")
    if iou_before is None or iou_after is None:
        return Decision("refuse" if mode == "preserve" else "counterfactual", mode, iou_before, iou_after,
                        None, tau, "input-view agreement not measurable")
    drop = iou_before - iou_after
    contradicts = drop > tau
    if mode == "counterfactual":
        return Decision("counterfactual", mode, iou_before, iou_after, drop, tau,
                        "labelled counterfactual" + (" (contradicts the photo)" if contradicts else ""))
    if contradicts:
        return Decision("refuse", mode, iou_before, iou_after, drop, tau,
                        f"fitted input-view IoU drops {drop:.3f} > tau {tau:.3f}: contradicts the photo")
    return Decision("accept", mode, iou_before, iou_after, drop, tau, "consistent with the photo")


def tau_from_appearance_arms(drops):
    """TAU = largest IoU drop among appearance-only edits (list of floats); floor 0.01."""
    return max(0.01, max(drops)) if drops else 0.05
