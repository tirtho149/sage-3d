# Expert review send-out kit (review item C1)

Status: **ready to send; raters not yet recruited** (only a person can do this step).
Protocol: `reports/13_expert_review_protocol.md`. Package: `SAGE3D_rater_package_v1.zip`
(88 files: README_for_raters.md, index.html, items/, rating_sheet.csv; verified to contain
no answer key). The key (`../key_DO_NOT_SHARE.csv`) stays with the coordinator.

## 1. Who to invite (protocol §4)

3 raters (a 4th if available), each meeting all of:
- PhD/MSc in plant pathology, or ≥ 3 years of diagnostic-clinic or extension experience;
- hands-on familiarity with at least two of: Peach Leaf Curl, Tomato Leaf Mold, Corn Smut;
- no involvement in this project and no knowledge of the editing methods;
- self-reported normal colour vision; a colour display ≥ 13 in, browser zoom 100%.

Possible sources (to be decided by the PI): ISU Plant and Insect Diagnostic Clinic, ISU
Plant Pathology faculty/students outside the project, extension plant pathologists.

## 2. Invitation email (draft — edit before sending)

> Subject: Request: ~1 hour blinded rating of plant-disease images (research study)
>
> Dear Dr. …,
>
> We are evaluating computer-generated images of three plant diseases (Peach Leaf Curl,
> Tomato Leaf Mold, Corn Smut) and need independent expert judgement of how realistic and
> disease-correct they are. The task is a blinded rating of 84 image pages in a web page
> that runs offline in any browser, taking about one hour. You would not need to know
> anything about how the images were made, and we ask that you work alone and not discuss
> items until all ratings are returned.
>
> If you are willing, we will send a zip file with instructions and a rating sheet.
> You will be credited in the acknowledgements (or as you prefer).
>
> Thank you,
> Tirtho Roy, Iowa State University (on behalf of Prof. Soumik Sarkar's group)

## 3. Coordinator checklist

- [ ] PI approves the rater list (qualifications above) and the email text.
- [ ] Assign codes R01–R04; keep the code ↔ name map separately (not in this repository).
- [ ] Send the zip + README only; never the key, manifest or build files.
- [ ] Optional 10-min calibration call using public photos that are not in the package.
- [ ] Collect `rating_sheet.csv` from each rater; save as `ratings/R0X.csv`.
- [ ] Run the pre-registered analysis, unchanged:
      `python src/review/agreement.py analyse --key results/review_package_v1/key_DO_NOT_SHARE.csv --ratings ratings/ --out results/review_package_v1/analysis`
- [ ] Report agreement first (Krippendorff's alpha per criterion); only criteria with
      adequate agreement go to the method comparison. Primary outcome: `f_overall_accept`,
      Holm over 3 comparisons (fixed before any rating).
