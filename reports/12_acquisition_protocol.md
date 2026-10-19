# 12: Acquisition protocol for falsifiable validation of KB-guided 3D edits

Status: protocol draft, not yet executed. Report 11 found no public data that can validate the underside, curl or gall claims, so this capture is the only route to an external test. All thresholds and success criteria in Section 9 must be frozen in the study's `protocol.yaml` / `hypotheses.jsonl` **before** the first specimen is reconstructed.

## 1. Claims under test and their null models

Each claim is stated so that it can fail. Each one is compared against the simplest 2D or flat alternative, because the advisor's question is what 3D adds over the photograph or a 2D edit.

| ID | Claim (what KB-guided 3D should add) | Null / 2D baseline it must beat | Ground truth from this capture |
|---|---|---|---|
| H-U | Underside symptom placement from the KB text (e.g., Tomato Leaf Mold: "olive-green to brown velvety mold forms on the lower leaf surface below leaf spots") matches the real abaxial surface of the same leaf. | B-U1: mirror the adaxial lesion mask onto the underside (a 2D rule that needs no 3D). B-U2: uniform random placement at the same area fraction. | Abaxial photograph of the same leaf, registered to the adaxial view, with a symptom mask. |
| H-C | The curl direction and magnitude produced by the KB geometry operator (Peach Leaf Curl: "distorted", "puckered", "thickened"; curl/roll operator) match the measured 3D shape of the same leaf. | B-C1: SAM 3D asset without the KB edit. B-C2: flat billboard (input image on a plane). B-C3: healthy-leaf mean curvature from controls. | MVS surface reconstructed from all non-input views, scaled by ChArUco. |
| H-G | Gall count and volume on the 3D asset (Corn Common Smut) match physical measurements. | B-G1: 2D area-based estimate (gall mask area^1.5 × constant fitted on controls; no 3D). B-G2: unedited SAM 3D asset. | Caliper axes, water-displacement volume, gall count. |
| H-V | The 3D asset reproduces held-out views better than a flat billboard. This is a precondition for H-U, H-C and H-G. | B-V: flat billboard at the input depth. | Real photographs at held-out azimuths and elevations. |

If a claim does not beat its 2D baseline, the result is recorded as negative or inconclusive (`arp/states.py::Failure`). It is not dropped.

## 2. Specimens and sourcing (Iowa State University, Ames, IA)

### 2.1 Target set

The target is 30 specimens. The minimum viable set is 20.

| Code | Class (SAGE) | Organ | Diseased (target / min) | Healthy control (target / min) | Primary claim |
|---|---|---|---|---|---|
| PLC / HPE | Peach Leaf Curl (*Taphrina deformans*) | leaf | 8 / 6 | 2 / 1 | H-C (curl, pucker, thickening) |
| CSM / HCO | Corn Common Smut (*Ustilago maydis*) | ear, stalk or tassel gall | 8 / 6 | 2 / 1 | H-G (gall count, volume) |
| TLM / HTO | Tomato Leaf Mold (*Passalora fulva*) | leaflet | 8 / 6 | 2 / 1 | H-U (underside) |

One specimen is one detached organ: one leaf, one leaflet, or one ear with its galls. If one plant yields several specimens, record `plant_id`. All statistics are clustered by plant (Section 8), so leaves from the same tree are not counted as independent.

### 2.2 Seasonal windows and sources

As of October 2026, the earliest feasible windows are:

| Disease | When symptoms are present in central Iowa | Sources to contact |
|---|---|---|
| Peach Leaf Curl | Spring, as new leaves expand after bud break. Infection occurs at bud swell, so symptomatic leaves cannot be collected in autumn. The next window is about April–May 2027. ISU Extension "Yard and Garden" lists Peach Leaf Curl as a problem in Iowa. | ISU Horticulture Research Station (Gilbert, IA) and backyard trees through ISU Extension and Master Gardeners. Samples submitted to the ISU Plant and Insect Diagnostic Clinic (PIDC) can confirm the diagnosis. |
| Corn Common Smut | Mid-summer to early autumn, from silking to ear fill (about July–September). Sweet corn is often more affected than field corn. | ISU agronomy research farms, sweet-corn growers and farmers' markets (smut galls are sometimes sold as huitlacoche), and PIDC. |
| Tomato Leaf Mold | A disease of greenhouses and high tunnels, favoured by high humidity. In Iowa it is seen mainly in high tunnels in summer and autumn. It can occur in greenhouses at other times. | High-tunnel growers through ISU Extension, ISU greenhouses (check with the greenhouse manager), and PIDC. |

**No inoculation is planned.** All three pathogens occur naturally in Iowa, and field or greenhouse samples suffice. If natural Tomato Leaf Mold cannot be found, any deliberate inoculation would need prior approval from the ISU greenhouse manager and the Institutional Biosafety Committee / Environmental Health and Safety, and is outside this protocol. Moving live plant pests across state lines can require a USDA APHIS PPQ 526 permit, so keep all collection in-state.

Every diseased specimen needs a diagnosis confirmation: either PIDC, or a plant pathologist's visual diagnosis plus a microscope check of signs (asci for *Taphrina*, teliospores for *Ustilago*, conidiophores for *Passalora*). Record the confirmation in `diagnosis`.

### 2.3 Handling and biosafety

- Collect into labelled sealed zip bags with a damp paper towel and keep at 4–8 °C. Capture within **6 h** for leaves and **3 h** for smut galls, which dehydrate and rupture. Record `minutes_collection_to_capture`.
- Mature smut galls release teliospore dust, and the spores are a respiratory irritant. Capture them inside a closed light tent or in a fume hood, wear nitrile gloves and an N95 respirator, and keep immature (intact, silvery) galls for 3D work.
- Do not bring sporulating Tomato Leaf Mold material into greenhouses that hold tomato research plants. Capture in a lab room that is not a growth facility.
- After capture, wipe surfaces with 70 % ethanol. Double-bag specimens and dispose of them by autoclave or as the department directs.

## 3. Capture rig

The rig is a single-camera turntable inside a diffuse light tent. A ChArUco board fixes scale and camera poses. Each specimen is held on a raised mount so that the camera can see the underside.

| Item | Specification | Purpose |
|---|---|---|
| Camera | Option A: recent smartphone in pro mode with RAW (DNG), fixed focus and exposure, main 1× lens (no wide lens). Option B: APS-C or full-frame body with a 50–60 mm macro lens at f/8–f/11. | Option B is preferred for depth of field on curled leaves. Use one option for the whole study. |
| Mount | Tripod with an arm that adjusts elevation, plus an angle gauge or the phone's level app. | Repeatable elevation rings |
| Turntable | Manual turntable marked every 15°, or a motorised one with 15° steps. | Repeatable azimuths |
| ChArUco board | 7×5 squares, square 20 mm, marker 15 mm, DICT_5X5_100, printed at 600 dpi on a matte rigid board. Measure the printed square with calipers and record it. | Intrinsics, poses and metric scale |
| Specimen mount | A thin black-painted rod (3 mm) set in the turntable centre, raising the specimen 80–120 mm above the board. Leaves are held at the petiole with a small clip. Ears are held on a spike through the shank. | Lets the −25° ring see the underside |
| Lighting | Light tent (60 cm), two LED panels of CRI ≥ 95 at 5000–5600 K at ±45°. If specular glare on waxy leaves is severe, use cross-polarisation (linear polariser on the light, circular polariser on the lens). | Uniform, repeatable colour |
| Colour reference | X-Rite / Calibrite ColorChecker Classic or Passport, placed in the first frame of every ring. | Colour correction |
| Scale bar | 50 mm ABFO-style scale or a steel rule, in the SAGE-like input view. | Independent scale check |
| Background | Matte mid-grey card behind the specimen. | Masking |
| Measurement | Digital calipers (0.01 mm), 0.01 g balance, 100 mL graduated cylinder or beaker plus an overflow cup. | Gall size and volume, leaf thickness |
| Flatbed scanner (optional) | 600 dpi, reflective mode. | High-resolution adaxial and abaxial maps |

Camera settings are fixed for the whole study: ISO ≤ 200, manual white balance set on the ColorChecker grey patch, manual focus at specimen centre, and RAW plus JPEG. Hold the focal length fixed (no zoom changes), because intrinsics are calibrated once per session from ChArUco frames.

## 4. Views per specimen

### 4.1 Turntable rings

There are 24 azimuths per ring (15° steps), and 0° is the specimen's "front".

| Ring | Elevation | Views | What it gives |
|---|---|---|---|
| R1 | +60° | 24 | Oblique adaxial (SAGE-like) |
| R2 | +30° | 24 | Side profile, which shows curl and roll |
| R3 | 0° | 24 | Silhouette and edge thickness |
| R4 | −25° | 24 | **Underside (abaxial) views** |
| T | +90° (top-down) | 1 | Adaxial nadir view |
| B | −90° (bottom-up) | 1 | Abaxial nadir view, shot from below through the mount clearance, or after rotating the clip by 180° |
| C | calibration | 6 | Board only, at varied angles, for intrinsics |

That gives 98 specimen views plus 6 calibration frames. For leaves, each ring is shot so that the leaf blade faces the camera at azimuth 0°. For ears with galls, set azimuth 0° to face the largest gall.

### 4.2 Input view and held-out views

The split must be fixed before any reconstruction.

- **Input view (I).** This is the one image the pipeline receives, chosen to mimic SAGE: **R1, azimuth 0°** (oblique adaxial), which includes the scale bar. For the SAGE flat-leaf style, a second input variant, **T** (top-down), is also evaluated and reported separately. The pipeline (SAM 3D Objects → KB edit) sees only I. No other view, mask or measurement may enter the reconstruction or editing.
- **Held-out evaluation views (E).** All other specimen views, grouped by angular distance from I:
  - **E-near:** ≤ 30°.
  - **E-mid:** 30–90°.
  - **E-far:** > 90°.
  - **E-under:** all R4 views plus B.
- **Reference reconstruction (Ref).** COLMAP SfM, then MVS (or a 3DGS fit at reference quality), using all 98 views except a **reference-check subset** of 8 views (R2 at 45°, 135°, 225° and 315°; R4 at 0°, 90°, 180° and 270°). Ref is scaled with ChArUco. The check subset is used only to measure the error of Ref itself, which sets the noise floor below which no method difference is interpretable.
- Every view's role is written to the metadata (`role` field) and hashed into the study registry before any reconstruction runs.

### 4.3 Both-surface photography of the same leaf (leaves only)

This step comes after the turntable capture, because it flattens the leaf.

1. Before flattening, pierce three registration pin holes (0.5 mm, a non-collinear triangle near the margin, away from lesions). The holes appear on both surfaces and give a fixed point correspondence.
2. Photograph the adaxial surface flat under 3 mm anti-reflective glass, top-down, with the ColorChecker and scale bar. Then turn the leaf over and photograph the abaxial surface the same way. Optionally scan both surfaces at 600 dpi.
3. Register abaxial to adaxial with a mirror transform plus a similarity or thin-plate-spline fit to the 3 pin holes and the leaf outline. Record the residual at the pin holes in mm. A residual above 1.0 mm marks the pair as `registration_failed`; the pair is kept and flagged.

### 4.4 Physical measurements

**Leaves (all).** Measure:
- blade length and width (mm);
- thickness at 3 points (calipers, mm). For Peach Leaf Curl, take two points on symptomatic tissue and one on the least-affected tissue.
- fresh mass (g);
- curl direction by visual scoring before flattening, against these categories: `adaxial_concave` (margins toward the upper side), `abaxial_concave` (margins rolled under), `mixed`, `none`;
- roll axis: `longitudinal` / `transverse` / `both` / `none`.

**Peach Leaf Curl, extra.** Count blisters (puckers) per leaf.

**Corn Smut.** Measure each gall separately:
- organ location (ear / shank / stalk / tassel / leaf) and maturity (`immature_white`, `silvery`, `ruptured_black`);
- three orthogonal caliper axes a ≥ b ≥ c (mm);
- after the 3D and flat photographs, excise the gall and record its mass (g) and **water-displacement volume** (mL). Use 3 repeats and record all three.

Also record:
- total gall count per ear, counting galls ≥ 3 mm;
- for intact ears: ear length and maximum diameter.

Ellipsoid volume πabc/6 is a secondary estimate. Water displacement is the reference.

## 5. Annotation

Annotation is done by two annotators. One of them is a plant pathologist or a trained plant-pathology student. A subset of 6 specimens is annotated by both annotators, and inter-annotator IoU is reported. Annotators do not see model outputs.

| Annotation | On which images | Tool / format |
|---|---|---|
| Specimen mask | All specimen views (semi-automatic segmentation, human-corrected on E and I) | PNG, 1 = specimen |
| Adaxial symptom mask | Flat adaxial photo; I; T | PNG, per-class labels: 1 lesion/chlorosis, 2 sporulation/mold, 3 blister/pucker, 4 gall |
| Abaxial symptom mask | Flat abaxial photo; B; one R4 view chosen by the annotator as the clearest | same label set |
| Curl label | Visual score at capture (Section 4.4) plus a second blind score from turntable views | JSON enum |
| Gall instances | Corn: polygon per gall on I and the flat photo, linked to `gall_id` | COCO-style JSON |
| Midrib polyline | Leaves: on T, for the curvature profile | JSON polyline |

Annotation masks are frozen and hashed before any model output exists for that specimen.

## 6. Metrics computed at held-out views and against measurements

All 3D comparisons first place the asset into the Ref frame **using only the input-view camera**: ChArUco pose of I, plus the scale from the scale bar seen in I. This is the "no-cheat" alignment. A second number, after a 7-DoF similarity ICP to Ref, is reported for context only. Claims are judged on the no-cheat number.

### 6.1 H-V (precondition, all specimens)

| Metric | Definition |
|---|---|
| Silhouette IoU | Render the asset mask at each E view (calibrated camera) and compute IoU with the annotated specimen mask. Report the median per specimen for E-near, E-mid, E-far and E-under. |
| Depth error | Absolute relative depth error and RMSE (mm) of the asset depth versus the Ref depth, on pixels inside both masks. |
| Chamfer distance | Symmetric mean (mm) between asset surface samples and the Ref point cloud. Also report F-score at 1 mm and 2 mm. |
| Photometric | Masked PSNR, SSIM and LPIPS at E views, after ColorChecker correction. Reported, but not used for decisions, because lighting differs between views. |
| Ref noise floor | The same metrics for Ref against the 8 reference-check views. |

### 6.2 H-U (underside; mainly TLM, also PLC and CSM where abaxial symptoms are visible)

| Metric | Definition |
|---|---|
| Underside location IoU | Render the predicted abaxial symptom mask from the asset at view B (and the chosen R4 view). Map both the rendered and the real abaxial masks into the flat-abaxial frame with the pin-hole registration, then compute IoU there. |
| Dice and boundary F1 (2 mm tolerance) | Same frame. |
| Centroid distance | Distance between matched lesion centroids (Hungarian matching), in mm and as a fraction of leaf length. |
| Area fraction error | Predicted minus real abaxial symptomatic area fraction. |
| Baselines | B-U1 (mirrored adaxial mask) and B-U2 (random placement, 100 draws, report mean), scored with the same metrics. |

H-U is supported only if 3D beats B-U1. If underside mold simply lies under the upper spots, B-U1 will do as well as 3D, and the honest conclusion is that 3D adds nothing for underside placement on that disease.

### 6.3 H-C (curl / 3D shape; mainly PLC, plus all leaves)

| Metric | Definition |
|---|---|
| Curl direction agreement | Fit a quadric surface (local scale 5 mm) to the asset and to Ref on the blade. Compute the sign of the mean transverse curvature κ_t across the midrib, relative to the adaxial normal. Agreement is the fraction of leaves whose predicted sign matches the Ref sign. The Ref sign is also checked against the visual curl label. |
| Curvature error | Absolute error in median κ_t and median κ_l (longitudinal), in mm⁻¹. Also compare the curvedness distribution √((κ1²+κ2²)/2) using the Wasserstein-1 distance. |
| Rolled-upness | Ratio of cross-section chord length to arc length at 25 %, 50 % and 75 % of blade length (after the Plant Methods 2015 "rolled-upness" idea); report the absolute error. |
| Pucker density | Count of local maxima of mean curvature above a threshold (set on healthy controls) per cm², compared with the blister count. |
| Thickness | Asset blade thickness (if represented) versus the caliper reading. Report "not represented" if the asset is a single sheet; this counts as a failure for "thickened". |
| Baselines | B-C1 (unedited), B-C2 (flat), B-C3 (healthy mean). |

### 6.4 H-G (gall volume; CSM)

| Metric | Definition |
|---|---|
| Count agreement | Absolute count error and Lin's concordance correlation coefficient (CCC) across ears. |
| Volume error | log(V_pred / V_water) per gall: report mean (bias) and SD. Also report relative error in % and CCC. |
| Scale-free variant | Gall volume divided by intact ear volume (Ref). This removes errors in absolute scale. |
| Baselines | B-G1 (2D area^1.5 fitted on 2 healthy plus a leave-one-out on diseased specimens), B-G2 (unedited asset). |

## 7. Metadata, file naming and storage

The machine-readable schema is `configs/acquisition/specimen_metadata_schema.json` (JSON Schema 2020-12). There is one `specimen.json` per specimen, plus `views.csv` with one row per image, whose columns mirror `views[]` in the schema.

Directory layout:

```
data/acquisition/<specimen_id>/
  specimen.json
  views.csv
  raw/         <specimen_id>_<ring>_a<azimuth:03d>.dng|.jpg     e.g. PLC-003_R4_a090.dng
               <specimen_id>_T.dng, <specimen_id>_B.dng
               <specimen_id>_CAL_<k:02d>.dng
  flat/        <specimen_id>_flat_ad.jpg, <specimen_id>_flat_ab.jpg, *_scan600_{ad,ab}.tif
  masks/       <specimen_id>_<view>_specimen.png, <specimen_id>_<view>_symptom.png
  ref/         colmap/, ref_mesh.ply, ref_check_metrics.json
  measurements/ galls.csv (CSM), leaf.csv
```

Rules:
- Specimen IDs follow `^(PLC|CSM|TLM|HPE|HCO|HTO)-[0-9]{3}$`.
- Ring codes are `R1`–`R4`, `T`, `B` and `CAL`. Azimuth is zero-padded degrees.
- Raw files are never edited. Each file's SHA-256 is recorded in `views.csv`.
- `data/acquisition/` is outside `results/` and `src/`. Bulk images belong in the project's data store, not in git.

## 8. Sample-size rationale

The primary analysis for each claim is a paired comparison per specimen: the 3D method's metric minus the 2D or null baseline's metric on the same specimen. This uses a paired t-test, or a Wilcoxon signed-rank test if the differences are clearly non-normal, two-sided, with α = 0.05 and power 0.80. The minimum detectable standardized paired effect d_z (mean difference / SD of differences), computed with the noncentral t distribution:

| n (paired specimens) | MDE d_z (α = 0.05) | MDE d_z (α = 0.0167, Bonferroni over 3 claims) |
|---|---|---|
| 8 (one disease) | 1.16 | 1.46 |
| 10 | 1.00 | 1.23 |
| 20 | 0.67 | 0.79 |
| 24 | 0.60 | 0.71 |
| 30 | 0.53 | 0.63 |

Assumptions and interpretation:
- Specimens are independent after clustering by plant. If several leaves come from one tree, use the plant mean, which reduces the effective n.
- Differences are roughly normal. The Wilcoxon test needs about 5 % more specimens for the same power, assuming normality.
- **Pooled across diseases (n = 24–30)**, H-V (3D vs billboard on held-out silhouette IoU, all specimens) can detect a medium-to-large effect (d_z ≈ 0.55–0.6).
- **Within one disease (n = 8)**, H-U, H-C and H-G can detect only large effects (d_z ≥ 1.2). As an example: if the SD of the paired IoU difference is 0.10, a mean improvement of 0.12 IoU over B-U1 is detectable and 0.05 is not. A null result at n = 8 is therefore "inconclusive", not "no effect", unless the 95 % CI of the difference excludes the smallest effect of interest.
- **Curl direction (binary).** With 8 Peach Leaf Curl leaves, at least 7/8 correct is needed to reject chance (p = 0.5) at one-sided α = 0.05. With all 24 leaves (PLC + TLM + controls), at least 17/24 is needed. Even 9/10 correct gives a Clopper–Pearson 95 % CI of 0.56–1.00, so direction accuracy will be estimated only coarsely.
- **Smallest effects of interest**, to be frozen before capture:
  - silhouette IoU +0.05 over billboard at E-mid;
  - underside IoU +0.10 over B-U1;
  - κ_t sign agreement ≥ 0.75;
  - gall log-volume bias |mean| ≤ 0.2, i.e., within about ±22 %.
- With 30 specimens these thresholds work as **equivalence or non-inferiority bounds** reported with CIs. They are not tests that can establish a positive result with high confidence.

Increasing to 15 per disease (45 diseased plus 6 controls) would bring the per-disease MDE to about d_z ≈ 0.78, if a season yields that many specimens.

## 9. Pre-registered decision rules (to copy into `protocol.yaml`)

- **H-V supported:** median per-specimen silhouette IoU(3D) − IoU(billboard) at E-mid is > 0.05, and the lower 95 % CI bound is > 0. Otherwise, H-U, H-C and H-G are reported as "not interpretable" for that specimen class.
- **H-U supported:** paired underside IoU(3D) − IoU(B-U1) has a lower 95 % CI bound > 0 on TLM. If the CI includes 0, the result is **inconclusive**. If the upper bound is < 0.10, the result is negative: 3D adds no material underside information over the 2D mirror rule.
- **H-C supported:** sign agreement ≥ 7/8 on PLC, and the median |Δκ_t| of the KB-edited asset is below that of B-C1 and of B-C3 (paired, lower CI bound > 0).
- **H-G supported:** |mean log(V_pred/V_water)| ≤ 0.2, and the error is lower than B-G1's (paired).
- If Ref's own check-view error exceeds a method difference, that difference is reported as below the noise floor.

## 10. Time and cost

### 10.1 Per-specimen time (one operator, manual turntable)

| Step | Minutes |
|---|---|
| Unbag, ID, mount, ColorChecker and calibration frames | 8 |
| 4 rings × 24 views, plus T and B | 25 (manual) / 12 (motorised) |
| Pin holes, flat adaxial and abaxial photos (plus optional scans) | 8 |
| Calipers, mass, curl score; gall excision and displacement (CSM) | 7 (leaf) / 20 (ear) |
| Metadata entry and file check | 5 |
| **Capture subtotal** | **about 50–65** |
| Annotation (masks on about 6 images, both surfaces, galls) | 30–45 |
| SfM/MVS reference, unattended, run through the job executor (not on a login node) | 30–60 compute |

At about 1.5 person-hours per specimen, 30 specimens need about 45 person-hours, plus about 10 hours for collection trips and clinic liaison. Capture sessions should be batched by disease within its seasonal window.

### 10.2 Equipment cost (approximate 2026 US retail; not quoted from vendors, verify before ordering)

| Item | Approx. USD |
|---|---|
| Light tent + 2 high-CRI LED panels | 150–300 |
| Turntable (manual / motorised) | 20 / 150 |
| ColorChecker Classic or Passport | 100–150 |
| ChArUco board printing and mounting | 20–40 |
| Tripod with elevation arm | 60–150 |
| Digital calipers, 0.01 g balance, cylinder, overflow cup | 60–120 |
| Polarising films and filter (optional) | 40–120 |
| Anti-reflective glass, pins, bags, PPE | 40–60 |
| **Subtotal (using an existing phone or camera)** | **about 500–1,100** |
| Optional mirrorless or DSLR body + 60 mm macro lens, if none is available | 900–1,800 |
| Optional flatbed scanner, if none is available | 100–250 |
| PIDC diagnostic fees | per the current PIDC fee schedule × number of specimens submitted |

Specimens themselves should cost little or nothing: they come from field, research-farm or grower donations. Purchased sweet corn ears with smut are an option.

## 11. Known limitations of this design

- The input view is a controlled-light turntable photo, not a field photo like many SAGE images. Results bound performance under favourable conditions; they do not estimate performance on SAGE itself.
- Flattening for both-surface photos is done after 3D capture, so curl ground truth is not affected. The abaxial mask is measured on the flattened leaf, so H-U is scored in the flat frame, not on the curled surface.
- 8 specimens per disease supports only large-effect conclusions (Section 8). One season may yield fewer than 8 Peach Leaf Curl leaves on separate trees.
- Smut gall volume by water displacement requires excision. Measure after all imaging and accept that galls may lose spores.
- The protocol tests three diseases. It does not show that KB-guided 3D generalises to the other 632 SAGE classes.
