# 01 — Case Selection (Phase 1, SAGE-only)

Date: 2026-10-12. Data: SAGE image shards (`data/metadata/sage_shard_manifest.parquet`, 820,502 rows; 635 classes with source-grounded KB text). Machine-readable table: `data/metadata/candidate_cases.csv` (includes KB source URL, quote excerpt, pathogen and disease type copied from SAGE, not from memory).

## Scope decision (user, 2026-10-12)

SAGE is the only data source. SAGE images are single, independent views with no specimen IDs, so **no specimen-matched multi-view or 3D ground truth exists**. Consequences:

- **H4 (hidden-trait recovery) and H1 on unseen views are out of scope.** Every property a 3D asset shows outside the input view is **"source-supported, unobserved"** at best. The KB says the disease can show it; nothing says this specimen does.
- The selection therefore asks a narrower question: **for which SAGE classes is the KB-described symptom intrinsically geometric or occluded?** These are the classes where a 3D asset could carry information that a 2D edit cannot represent. Whether that information is *useful* is tested downstream (H5) on real SAGE test images.

## Method

1. **Keyword screen.** Ran a regex screen of every KB class's `visual_symptoms` and verbatim `symptom_quote` for seven 3D-relevant symptom families: curl/roll, gall/knot, canker, underside, distortion, witches' broom, and root (`data/metadata/sage_3d_keyword_hits.csv`, 313 class-hits).
2. **Check against the source text.** Read the hit clause in the source text, not just the keyword. Mentions where the term is incidental were discarded.
3. **Check the actual images.** Ten candidate classes were extracted (byte-level dedup, ≤300 per class) and inspected as contact sheets (`figures/supplementary/contact/`). A class qualifies only if its photos actually show an isolable organ.
4. **Label the condition kind.** Every class here is labelled from the SAGE `disease_type` field. All selected classes are diseases (fungal or bacterial). No insect-pest or abiotic classes are included. Tea tortrix and root-knot nematode, named in the brief, are pests and are **not present in SAGE**.

## Selected cases

| Case | Class (SAGE) | Type (SAGE) | Why 3D could matter (KB, verbatim source) | What the input view lacks | Unique images |
|---|---|---|---|---|---|
| **A — geometry-dominant** | Peach / Leaf Curl | Fungal (*Taphrina deformans*) | "leaves become severely distorted (thickened and puckered)" (Ohioline PLPATH-FRU-26) | Pucker depth and the far side of a curled leaf | 222 |
| **B — occlusion-dominant** | Tomato / Leaf Mold | Fungal (*Passalora fulva*) | "olive-green to brown velvety mold forms on the **lower leaf surface**" (UMN Extension) | SAGE photos of this class are top-down lab shots, so the lower surface is never observed | 16,578 |
| **C — volumetric outgrowth** | Corn / Smut | Fungal (*Ustilago maydis*) | "tumor-like galls that vary from less than 1 cm to more than 30 cm" (APSnet) | Gall volume and the back of the ear | 505 |
| **D — surface relief** | Citrus / Canker | Bacterial (*X. citri* ssp. *citri*) | "becoming corky with a crater-like depression" (CDFA) | Relief under single lighting; the far hemisphere | 547 |
| **Z — negative control** | Mango / Anthracnose | Fungal (*C. gloeosporioides*) | Flat dark spots on a near-planar leaf | Nothing material | 2,600 |

The negative control is essential. If KB-guided 3D shows a "benefit" on Mango Anthracnose as large as on cases A–D, the benefit is not coming from geometry.

## Deferred / excluded (with reasons)

| Class | Decision | Reason |
|---|---|---|
| Grape / Leafroll | Deferred | Strongest geometric KB (every cited clause is geometric), but images are vineyard canopy scenes; no isolable leaf. |
| Cherry, Plum / Black Knot | Deferred | 40 unique images in total (21 + 19 after dedup), mostly whole trunks. |
| Squash / Downy Mildew | **Excluded, label-quality flag** | Many images show a white powdery coating on the *upper* surface. That is characteristic of powdery mildew, not downy mildew (whose KB says grey fuzz on the underside). Needs expert review of SAGE labels. |
| Pumpkin / Mosaic | Excluded | Field canopy shots; includes edge-filtered and grayscale "processed" images from the source dataset. |
| Root diseases | Excluded | Only 167 root-keyword images across all of SAGE. |

## Data-quality findings surfaced during selection

- **Exact-duplicate images:** e.g. Cherry Black Knot 40 → 21 unique files. The shard manifest has 16,973 duplicate (crop, disease, filename) rows. All splits must deduplicate by content hash before partitioning.
- **Processed/grayscale images:** some source datasets (e.g. A2H0H0R1Plan, Pumpkin_Leaf) contain processed or grayscale images. An automatic screen (`sat > 0.12`, min side ≥ 200 px; column `auto_ok` in the specimen manifest) removes most of them. The screen also wrongly rejects some white-background Mango images (193/300 pass), which is acceptable for the pilot and noted as a limitation.
- **Possible label noise:** Squash Downy Mildew (above).

## Pilot specimens

Three per selected class (15 total), listed in `experiments/configs/pilot.yaml`. They were chosen **before any model was run**, against fixed criteria: single dominant organ, organ mostly in frame, colour photo, no text over the organ, not a collage. Object masks come from SAM ViT-H with an annotator box prompt (`src/reconstruction/make_masks.py`). This is the object-selection input that SAM 3D itself requires.
