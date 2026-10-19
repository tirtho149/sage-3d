# 17 — Response to the detailed scientific review of the SAGE-3D deck

Review received 2026-10-15 ("Major scientific revisions needed before publication").
Work log: `REVIEW_TODO.md`. Deck: Overleaf 6ac800709ea44535f867ab26 — full deck `main.tex`
(41 slides, PDF `reports/final/SAGE_3D_deck_v5.pdf`) and 10-slide meeting deck `meeting.tex`
(`reports/final/SAGE_3D_meeting_v5.pdf`). Every slide number is generated from `results/`;
`experiments/review/b10_consistency.py` regenerates all 266 macros and checks the deck.

## Summary of what changed the conclusions

1. **Consistency is not evidence for our editor.** A 2D KB edit painted onto the shared asset
   is as consistent (cross-view dE 2.7, floor 2.0) as the geometry editor while recolouring
   more (dE 32). The E4 advantage comes from sharing one asset (B6).
2. **The photo constrains edits.** An explicit preserve / counterfactual / abstain policy,
   with τ = 0.046 from appearance-only edits: pucker and two-surface edits pass; every curl
   (0/22 on 11 leaves, 0/18 on Mango) contradicts its photo; KB-sized galls do too unless
   placed with a render check, which forces smaller galls (B2, B3, B1).
3. **3D beyond 2D is real but partial.** Direction: the best-case 2D warp barely separates
   up from down (dE 5–14 vs 41–46 in 3D, B7). Surface: a two-layer 2D image is wrong only on
   0–23% of leaf pixels, so the gain is small for flat leaves (B4).
4. **The curl operator does not transfer to every leaf.** Direction holds on 11/11 leaves,
   magnitude varies 7–43° for one target, and it fails on 2 Citrus leaves (B1).
5. **The design cannot detect KB effects.** With 4 informative specimen pairs the exact
   test cannot go below p = 0.125 (Holm ≥ 0.875) at any effect size; only artefact measures
   can survive correction (B5). More disease classes are needed, not more seeds.
6. **The downstream failure is a rendering problem.** Renders are trivially separable from
   photos even on white backgrounds; the matched pilot attributes the gap to object
   rendering (+0.28) and novel viewpoints (+0.21), not colour (−0.01) (A12, B8).

## Point-by-point

| Review point | Response | Evidence |
|---|---|---|
| §3 Exp. 1: mango curl is counterfactual | Labelled as a positive control on the slide; preserve policy refuses it; curl tested on all 12 leaves with independent measures (profile, strain, self-contact, midrib) | slide "Test 1"; B1; B2 |
| §3 Exp. 1: direction across plant shapes | 11/11 leaves follow direction; magnitude 7–43°; fails on 2 Citrus leaves | B1 |
| §3 Exp. 1: constraint against contradicting the photo | `evidence_policy.py` (preserve / counterfactual / abstain) | B2 |
| §3 Exp. 2: cm³ uncalibrated | Scale-free volume (‰ of ear length³) is primary; cm³ labelled assumed | A2 |
| §3 Exp. 2: kernel placement 1.00 is by construction | Labelled constraint compliance; kernel vs random placement contradict the photo equally | A2, B3 |
| §3 Exp. 2: variable counts/severity; generic volumetric editor | n = 1/3/6 sweep; random-site arm as matched generic placement; render-checked placement | B3 |
| §3 Exp. 2: real smut galls | Size plausibility vs galls visible in the photos (0.09–0.14 of ear length); real measured galls still needed | B3; C3 (blocked) |
| §3 Exp. 3: two-layer 2D baseline | Measured what any two-layer 2D image must get wrong: 0–23% of leaf pixels | B4 |
| §3 Exp. 3: underside copied from upper mask? | Yes, the footprint is copied by construction (KB "below leaf spots"); only texture differs. Stated | B4 |
| §3 Exp. 3: thickness | Assumed 0.004 L, stated; not measurable without specimens | B4 |
| §4A circular evaluation | Three evaluation levels on the slides; recall labelled compliance | A7, report 16 |
| §4B weak 2D baseline | Best-case 2D warp (exact projection of the 3D curl) | B7 |
| §4C coherence partly inherent; show edit magnitude | Edit size beside consistency; shared-texture baseline | A6, B6 |
| §4D fitted IoU ≠ accuracy | "Fitted input-view IoU" everywhere; 3D accuracy stated as not measurable | A5 |
| §4E methodological novelty | Objective G* with L_obs as a hard constraint; terms mapped to code; ablation; limits | B9, report 16 |
| §5 statistics | Paired per-specimen values; leave-one-class-out; exact-test floors explain CI vs Holm; primary outcome fixed (reports/13) | B5 |
| §6 negative downstream | Error analysis + matched gap decomposition before any 250 GPU-h study | A12, B8 |
| §7 slide-level corrections | Slides 17/18/20/23/26 fixed; deck reordered; failures mapped to fixes; streams labelled | A1–A11 |
| §8.1 expert validation | Send-out kit ready (zip without key, invitation draft, checklist) | C1 (needs raters) |
| §8.2 specimen-matched multi-view | Protocol exists (reports/12); needs specimens | C2 (blocked) |
| §8.3 independent geometry evaluation | Operator-independent measures (B1); real reference geometry still needed | B1; C3 |
| §8.4 render-to-real diagnosis | Done as a small matched pilot | B8 |
| §8.5 stronger baselines | Best-case 2D warp, two-layer 2D, shared-texture 2D | B4, B6, B7 |
| §12 compact presentation | `meeting.tex`, title + 10 slides in the review's order | A13 |

## Still open (needs people or specimens)

- **C1** blinded plant-pathologist ratings (kit: `results/review_package_v1/sendout/`).
- **C2** specimen-matched multi-view capture (protocol: `reports/12`).
- **C3** measured corn smut galls, real curl phenotypes, paired upper/lower photographs.
