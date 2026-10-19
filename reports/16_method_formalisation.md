# 16 — The geometry editor as an evidence-conditioned constrained objective

Response to review §4E ("the paper needs a stronger methodological novelty argument").
This report states the editor as one objective, maps every term to the code that
implements it, and says plainly what is **not** implemented. Ablations of the existing
terms: `results/review/b9_ablation/summary.md`.

## 1. Objective

For an asset with Gaussian centres X (from one photo), a KB statement k and the photo's
organ mask M, the editor solves

    G* = argmin_G   λ_KB · L_KB(G; k) + λ_anat · L_anat(G; X) + λ_reg · L_reg(G; X)
         subject to  L_obs(G; M) ≤ τ          (preserve mode)
         or          no constraint, output labelled counterfactual   (counterfactual mode)
         or          no edit if k does not specify the attribute    (abstain)

| Term | Meaning | Implementation | Status |
|---|---|---|---|
| L_KB | the edit realises a source-supported attribute (direction, relief, outgrowth, surface) | `deform.py` L_edit with targets from `curl.curl_target` (signed roll), pucker/distort/outgrowth/depress RBF fields; `gall.build` (volumetric galls); `bilayer.build` (two surfaces); placement weights from `placement.py` (lesions, kernel tissue) | implemented |
| L_anat | tissue behaves like a leaf/ear: no stretching, organ-specific placement | `deform.py` L_arap (+ coarse ARAP), kernel-tissue placement for galls, adaxial/abaxial layer separation | implemented (generic tissue model; no species-specific mechanics) |
| L_reg | no unnecessary change | `deform.py` L_smooth (Laplacian of the displacement) and L_anchor (non-target parts stay) | implemented |
| L_obs | do not contradict what the photo shows | fitted input-view IoU drop vs the REAL mask, ≤ τ (τ = largest drop of appearance-only edits, 0.046) | implemented as a **hard check**: `evidence_policy.decide` (accept/refuse) and render-checked rejection sampling for galls (`experiments/review/b3_corn_galls.py`, arm Vp). Not a differentiable loss. |
| abstain | KB silent on an attribute → no edit | `curl.parse` (no direction → abstain), corn abstains without kernel tissue | implemented |

## 2. What is new relative to "a rule-based deformation script"

1. **Source grounding with explicit abstention.** Every operator parameter traces to a
   verbatim KB phrase or a logged assumption (`data/kb/supplementary_evidence.json`,
   per-edit logs); missing facts produce abstention or a labelled hypothesis, never a guess.
2. **Observed evidence as a constraint, not a metric.** The same fitted-IoU measure that
   reviewers use to check consistency is enforced before an edit is accepted; when the
   KB's typical size cannot fit this photo, the photo wins (B3: galls on Corn 0012 forced
   to eq. diameter 0.061 L instead of 0.094 L).
3. **Representation choices that a single image cannot make:** signed direction (B7: the
   best-case 2D warp barely separates up from down, dE 5–14 vs 41–46 in 3D), closed volumes
   with overhang, two surfaces.

## 3. Not implemented (stated so it is not over-claimed)

- **Uncertainty-aware constraints.** The KB rarely fixes counts, positions or sizes. The
  editor samples them (seeds; rejection sampling under L_obs) but does not represent a
  distribution over geometries or report its spread as an uncertainty of the output.
- **Differentiable L_obs.** The photo constraint is a post-hoc accept/reject check; it
  does not shape the optimisation.
- **Learned or species-specific anatomy.** L_anat is a generic elastic-sheet prior.
- **Automatic KB → operator mapping.** The phrase → operator table is hand-written; a
  learned or verified mapping is future work, and expert review of the mapping is pending.

## 4. Evidence status of each claim

| Claim | Level (review §4A) | Evidence |
|---|---|---|
| Operators follow the KB attribute | operator compliance | Test 1–3; B1 (all 12 leaves) |
| Edits do not contradict the photo | operator compliance (measured) | B2 policy, B3 Vp |
| 3D can express direction / volume / surface beyond 2D | representational | B4, B7 |
| Shapes are biologically realistic | biological fidelity | **not measured** (expert ratings pending, reports/13) |
| Useful for a downstream task | practical validity | **not supported** (E5; gap diagnosed in A12/B8) |
