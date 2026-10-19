# 13 — Expert-review protocol (blinded rating by plant pathologists)

**Status (2026-10-14): package built; no expert ratings exist.** Every number below that describes ratings comes from SIMULATED data and is shown only to check the analysis code. The biological validity of every arm remains unvalidated.

## 1. Purpose

Reports 03–10 measure geometric and appearance properties with automatic detectors, and all of those are internal measures. None of them shows that an edited asset looks like the disease to a pathologist. This protocol asks 3 plant pathologists to rate blinded candidates against the knowledge-base (KB) text that drove the edits. It answers two separate questions, in this order:

1. **Agreement.** Can independent experts apply the rubric consistently? If they cannot, no method claim follows from the ratings.
2. **Method effect.** Only for criteria with adequate agreement: do the KB-guided 3D edits get higher scores than the KB-guided 2D edits and the generic 3D edits?

## 2. Files

| file | role |
|---|---|
| `configs/review/rubric.json` | rubric, anchors, catch rule and pre-registered analysis settings (machine-readable) |
| `src/review/manifest_v1.py` | writes the v1 manifest from the files on disk |
| `src/review/build_package.py` | manifest → blinded package (order, ids, composites, form, sheet, README, sealed key) |
| `src/review/review_form_template.html` | offline rating form (static HTML+JS; autosave in the browser; CSV export and re-import) |
| `src/review/agreement.py` | agreement and method-effect analysis; `simulate` writes SIMULATED ratings for dry runs |
| `tests/test_review_agreement.py` | checks against textbook values (Krippendorff 2011; Shrout & Fleiss 1979; sklearn kappa) |
| `results/review_package_v1/manifest.csv` | v1 input (84 rows, 75 included) |
| `results/review_package_v1/key_DO_NOT_SHARE.csv`, `build_meta.json` | sealed key and build record; **do not send to raters** |
| `results/review_package_v1/rater_package/` | **the only folder sent to raters** |
| `results/review_package_v1/dryrun_SIMULATED/` | SIMULATED ratings and the analysis of them (pipeline check only) |

To add items (for example `results/demos/<demo>/<method>/...`), append rows to a manifest and rebuild into a new directory (`--out results/review_package_v2`). No code change is needed. The method family is taken from the `method_family` column, or from `rubric.json → analysis.families` when that column is blank.

```bash
PY=~/micromamba/envs/sam3d/bin/python
$PY src/review/manifest_v1.py                       # -> results/review_package_v1/manifest.csv
$PY src/review/build_package.py                     # seed 20261014, 12% duplicates, gap >= 20
$PY src/review/agreement.py analyse --key results/review_package_v1/key_DO_NOT_SHARE.csv \
    --ratings <dir of returned CSVs> --out results/review_package_v1/analysis --confirm-real
```

## 3. Package v1

**Scope.** Package v1 covers Peach Leaf Curl (0006, 0063, 0087), Corn Smut (0012, 0025, 0052) and Tomato Leaf Mold (0015, 0052, 0092), all at seed 0.

| arm | what it is (hidden from raters) | provenance | items |
|---|---|---|---|
| B | 2D generic edit | EDITED-REAL-2D | 9 |
| C_stage | 2D KB stage edit | EDITED-REAL-2D | 9 |
| I | 2D shape-capable edit | EDITED-REAL-2D | 9 |
| D | unedited 3D reconstruction | SYNTHETIC | 9 |
| E | GaussianEditor, generic prompt | SYNTHETIC | 9 |
| G | generic geometry | SYNTHETIC | 6 (none for Tomato) |
| J | KB geometry | SYNTHETIC | 6 (none for Tomato) |
| K | KB appearance, geometry frozen | SYNTHETIC | 9 |
| H | J then K | SYNTHETIC | 6 (no Tomato renders on disk) |
| catch | 2D edit prompted with a different disease (`C_mismatch`) | EDITED-REAL-2D | 3 |
| duplicates | re-presented test items (≥ 20 positions later) | — | 9 |
| **total** | | | **84** |

**Time and size.** At about 35 s per item plus 8 min for instructions, a session takes about **57 min**. The rater folder is 29 MB (84 PNG composites, the form, the sheet and the README).

**Composite layout.** Each item is one PNG. The REAL input photograph is always shown first, with a green label ("REAL photo (reference)"). The candidate panels have grey labels that read only "candidate". For 2D candidates, the candidate panel is the same size as the REAL panel. For 3D candidates, five rig views are shown at a common scale: front +20°, back face 180° +20°, side 90° +20°, from above +80°, from below −80°. The item page in `index.html` also shows the verbatim KB `source_quote`, the sentences of the stage being judged, and the sentences of other stages ("should not be present at this stage").

**Deviations from the original plan.**

- **View 4 replaces view 6.** The requested views were 0, 2, 6, 16, 17. The rig has 18 views: 8 azimuths at ±20°, plus views 16 and 17 at ±80° (`src/rendering/views.py::rig`, `n_az = 8`). For leaves, views 2 and 6 are edge-on, and views 16 and 17 look along the leaf axis, so none of the requested views shows the opposite face of the lamina. Criterion (d) needs that face, so view 6 (a mirror of view 2) was replaced by view 4 (azimuth 180°). In the inspected Tomato K item, view 4 shows the upper-surface spots repeated on the back face. A rater can judge this only with that view.
- **Arm F is excluded.** GaussianEditor + KB prompt is listed in the manifest with `include = 0`. Including it would add 9 items and about 6 min, which takes the session past 60 min.
- **Tomato has no G, J or H items.** The KB gives no geometry operator for Tomato Leaf Mold, so G and J were not run. H has only a claim file and no renders.
- **Corn has no catch item.** The Corn wrong-disease edits (prompted with northern leaf blight) are almost identical to the input photograph, which already shows galls. They would not test attention. The 3 catch items are Peach 0063, Peach 0087 and Tomato 0052, whose wrong-disease edits visibly remove the described symptoms.

**Blinding.** Raters see only:

- the anonymised item id (`X-` followed by 4 random characters);
- the disease name, the organ and the KB text;
- the composite image.

Method, arm, provenance, catch status and duplicate status appear only in `key_DO_NOT_SHARE.csv`. A scan of the rater folder found no arm names, path fragments or method words. **The modality cannot be blinded.** A single edited photograph and a five-view render are easy to tell apart, so 2D-vs-3D comparisons may carry an expectation bias that within-modality comparisons do not.

**Specimens are already diseased.** The input photographs already show the disease, so the unedited reconstruction D is a meaningful control. A KB edit has to reach at least the D score to be useful, and it must not reduce realism.

## 4. Raters

**Number.** We will use **3** raters, and recruit a fourth if one is available.

- Three is the smallest panel for which pairwise disagreement can be attributed to one rater: with 3 pairs, one deviant rater shows up in two low kappas.
- ICC(2,k) treats the raters as a random sample, which needs k ≥ 3 to be estimated with any meaning.
- Each item gets 3 pairable values, so the 72 primary items give 216 values per criterion. That is enough for a unit-bootstrap CI on alpha with a half-width of roughly ±0.1–0.15 (SIMULATED dry run: CI half-widths 0.13–0.17).
- A fourth rater mainly protects the design when one rater fails the catch rule. With 4 raters, an exclusion still leaves 3.

**Qualifications.**

- (i) Either a PhD or MSc in plant pathology, or at least 3 years of diagnostic-clinic or extension experience.
- (ii) Hands-on familiarity with at least two of the three diseases.
- (iii) No involvement in this project, and no knowledge of the editing methods.
- (iv) Self-reported normal colour vision, because several criteria depend on hue (reddish-purple cast, olive-brown sporulation).
- (v) A colour display of 13 inches or larger, with the browser zoom at 100%.

**Independence.** Raters work alone, do not discuss items until all sheets are returned, and receive only `rater_package/`. Each rater gets a code (R01–R04). The coordinator keeps the mapping from codes to names separately.

**Calibration.** The first three item pages show the anchors expanded. No calibration is done on package items. Optionally, the coordinator holds a 10-min call before rating, using public photographs that are not in the package, to go through the anchors.

## 5. Rubric

The full anchor text is in `configs/review/rubric.json`, and it is reproduced in the README and on every form page. Each scale runs from 1 to 5, and NA is allowed when the images cannot show the criterion. NA is treated as missing data, not as a low score.

| id | criterion | 1 means | 5 means |
|---|---|---|---|
| a_symptom_match | symptoms named in the KB text for this stage are present | no described symptom, or another disease | all named symptoms present, matching colour, texture and extent |
| b_distribution | location and distribution plausible for disease and organ (galls from kernels on the ear; olive-brown sporulation on the lower surface beneath upper-surface yellow spots; puckering spread over the lamina) | wrong organ or impossible pattern | as seen in field specimens |
| c_geometry | 3D form plausible (curl direction, pucker or blister form, gall volume and attachment, lesion depression) | torn, self-intersecting, spiky, or a shape change the disease does not cause | form indistinguishable from real specimens; unchanged where no shape change is implied |
| d_surface | symptoms on the correct surface (upper vs lower) and consistent across views | wrong surface, or contradictory surfaces | each surface carries exactly its described symptoms |
| e_artefacts | absence of non-biological artefacts (grain, colour drift, flicker, floaters, smear, seams) | artefacts dominate | none visible |
| f_overall_accept | would accept as a realistic example of this disease at this stage | definitely not; misleading | definitely yes; as good as a photograph |
| prohibited_feature | yes/no: a feature the KB places in another stage, another disease's symptom, a change to background or unaffected parts, or an undescribed shape change | — | — |
| comment | free text | — | — |

Intermediate anchors (2–4) are defined for every scale in `rubric.json`.

## 6. Pre-registered analysis

All settings were fixed in `rubric.json → analysis` before any rating. The analysis is `src/review/agreement.py analyse`.

**Unit of analysis.** The agreement statistics use the first presentation of the 72 test items. Duplicates and catch items are excluded from them.

### 6.1 Agreement

- **Krippendorff's alpha per criterion.** The ordinal metric is used for the 1–5 scales and the nominal metric for `prohibited_feature`, with missing values allowed. The 95% CI comes from a unit (item) bootstrap with 2000 replicates and seed 20261014.
- **Interpretation thresholds** follow Krippendorff (2004):
  - α ≥ 0.800 means reliable;
  - 0.667 ≤ α < 0.800 allows tentative conclusions;
  - α < 0.667 means that criterion supports no method claim.
- **Quadratic-weighted Cohen's kappa** for each rater pair, per criterion.
- **ICC(2,k)** (two-way random, average measures; Shrout & Fleiss 1979) on items that every rater scored. It is reported beside alpha because it treats the scores as interval data.
- **Intra-rater reliability.** For each rater and criterion, the 9 duplicate pairs give a quadratic kappa, the exact-agreement rate and the mean absolute difference. With 9 pairs these are descriptive only.
- **Catch items.** A rater passes a catch item if `a_symptom_match` ≤ 2 or `prohibited_feature` = yes. A rater who passes fewer than 2 of the 3 is flagged.
  - Flagged raters stay in the agreement statistics.
  - They are excluded from the main method-effect analysis.
  - A sensitivity analysis with all raters is always reported.

### 6.2 Method effect

The method-effect analysis is a separate section. It is reported only for criteria with α ≥ 0.667, and the other criteria are listed as uninterpretable.

**Item and specimen scores.** The item score is the mean over raters. The specimen is the independent unit, giving 9 specimens.

**Family means.** Family means are computed per specimen:

| family | arms |
|---|---|
| KB-3D | H, J, K (and F when included) |
| generic-3D | E, G |
| KB-2D | C_stage |
| generic-2D | B |
| shape-2D | I |
| unedited-3D | D |

**Planned paired comparisons.** These use the specimens that have both families. For each comparison we report:

- the mean difference;
- a specimen-bootstrap 95% CI;
- an exact two-sided sign-flip permutation p (2⁹ = 512 sign patterns, so the smallest attainable p is 0.0039).

| comparison | specimens |
|---|---|
| KB-3D vs KB-2D | 9 |
| KB-3D vs generic-3D | 9 |
| KB-2D vs generic-2D | 9 |

**Primary and secondary criteria.**

- The primary criterion is `f_overall_accept`, with Holm correction across the 3 comparisons.
- Criteria a–e are secondary. They are reported with unadjusted CIs and labelled exploratory.
- Per-arm means use specimen-clustered bootstrap CIs.
- Per-disease results are descriptive only, because each disease has 3 specimens.

**Reporting rule.** Every pre-registered statistic is reported, including null and negative results. No item or rater is removed after unblinding except by the catch rule above.

**Power.** Nine specimens can detect only large, consistent differences. The study is designed to estimate agreement and effect size, not to establish small effects. In the SIMULATED dry run, the KB-3D latent quality was set about 0.75 scale points above generic-3D, before rater noise (SD 0.6) and rounding. The resulting difference on `f_overall_accept` was 0.35 [0.11, 0.66], with Holm p = 0.059. This illustrates the low power; it is not a result.

## 7. Verification of the analysis code (no real data)

`tests/test_review_agreement.py` contains 14 tests, all of which pass:

- Krippendorff (2011), example C (4 observers × 12 units, missing data): nominal 0.743, ordinal 0.815, interval 0.849, ratio 0.797.
- Krippendorff (2011), binary example: 0.095.
- Perfect agreement gives α = 1, and independent uniform ratings give |α| < 0.03.
- Quadratic kappa equals 1 for identical ratings and −1 for reversed ratings. It matches `sklearn.metrics.cohen_kappa_score` to 1e-12, and gives ≈ 0 for independent ratings.
- The ICCs reproduce Shrout & Fleiss (1979), Table 4 (for example, ICC(2,1) = 0.29 and ICC(2,k) = 0.62).
- The Holm and sign-flip tests are checked.
- An end-to-end run on SIMULATED ratings checks that:
  - catch items are passed;
  - the planted KB-3D > generic-3D effect is recovered in direction;
  - every output file carries the "No real expert ratings exist yet" status.

The rating form was also run in headless Chrome. Its validation blocks incomplete pages, it saves and advances, and the CSV export (including quoted comments with line breaks) parses as expected.

## 8. Limitations

- **Unblinded modality.** 2D and 3D candidates are distinguishable by eye (§3).
- **Single seed and few specimens.** One seed and 3 specimens per disease mean that the per-disease conclusions are not testable.
- **The KB text is the standard.** A rater may consider the KB description incomplete. Comments are collected so that this can be separated from disagreement about the images.
- **Renders on white.** 3D renders appear on a white background, while REAL and 2D images keep their photographic background. This difference may affect criterion (e) independently of the edit.
- **Coverage.** Citrus Canker and Mango Anthracnose are not in v1. They can be added through the manifest.
