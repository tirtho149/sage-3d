# DiseaseNet: Large-Scale Synthetic-to-Real Plant Disease Classification

> Saved verbatim from the user's planning document (2026-10-15) as the master protocol for
> the next phase after SAGE-3D. Status: **plan only, nothing executed.** See
> `EXECUTION_PLAN.md` for how it maps onto the existing code, data and hardware.

Proposed research title: DiseaseNet: Knowledge-Grounded Synthetic-to-Real Learning for Scalable Plant Disease Recognition

This is a stronger and more focused direction for the next phase of SAGE-3D: use the 3D pipeline as a synthetic training-data engine, train a comparatively simple visual classifier, and evaluate whether the learned disease representations generalize to real SAGE images.

The project should have two distinct contributions:

1. DiseaseNet-Generator: Generate large-scale, source-grounded synthetic disease images with controlled morphology, disease stage, viewpoint, illumination, and background.
2. DiseaseNet-Classifier: Fine-tune a lightweight or medium-size ViT on synthetic images and evaluate it on unseen synthetic assets and real SAGE photographs.

The central hypothesis is:

> Can a classifier trained on knowledge-grounded synthetic plant disease images learn sufficiently transferable disease features to identify real plant diseases across hundreds of crop–disease classes?

## 1. Important research correction: IID, OOD, and SAGE

Use three distinct evaluation concepts.

| Evaluation | Training / evaluation distribution | Purpose |
| --- | --- | --- |
| Synthetic IID | Synthetic train → independently generated synthetic test | Measure learning and generalization within the synthetic domain |
| Synthetic-to-real domain shift | Synthetic train → real SAGE test, same crop–disease classes | Primary DiseaseNet experiment |
| Open-set OOD | Synthetic train → real SAGE images of classes absent from training | Measure unfamiliar-class detection, not ordinary closed-set accuracy |

Real SAGE images of the same classes should not automatically be called class OOD. They are out of domain because the image distribution changes; the labels remain in distribution.

Likewise, different renderings of one reconstructed 3D asset cannot be split between synthetic training and testing and treated as independent IID specimens.

### Proposed end-to-end pipeline

SAGE Image + Disease Knowledge Base (crop, disease, anatomical organ, symptoms, source evidence)
→ 3D Geometry Engine (SAM 3D, pose correction, anatomical deformation, controlled severity)
→ KB-Grounded Disease Engine (stage-specific symptoms, target organs, appearance and geometry constraints)
→ DiseaseNet Synthetic Dataset Factory (multiple source assets × morphology × viewpoints × lighting × backgrounds × severity)
→ Simple ViT / DINOv2 Classifier (image-only inference, fine-tuned on synthetic data)
→ Synthetic IID (unseen parent assets) · Real SAGE (same-class domain shift) · Open-set OOD (unseen disease labels)

## 2. What to borrow from InsectNet and WheatNet

**InsectNet (PNAS Nexus, 2025)** — https://academic.oup.com/pnasnexus/article/4/1/pgae575/7933354
InsectNet's transferable ideas are self-supervised representations, global-to-local fine-tuning, large-scale class coverage, long-tail evaluation, uncertainty detection, and conformal prediction. Its reported accuracy cannot be directly transferred to DiseaseNet because its dataset, supervision, and domain are different.

**WheatNet (Khaki et al.)** — https://arxiv.org/abs/2103.09408
WheatNet contributes the philosophy of efficient agricultural vision models, multi-scale feature extraction, and robustness to scale. Its published task is wheat-head detection and counting—not disease classification. The paper is by Saeed Khaki, Nima Safaei, Hieu Pham, and Lizhi Wang; Soumik Sarkar is not listed as an author.

For DiseaseNet, adopt InsectNet's representation learning and trustworthy evaluation, and WheatNet's lightweight, multi-scale architecture philosophy. The classifier itself can remain simple: a pretrained ViT with a linear classification head.

## 3. The major challenge identified by the current experiment

The latest SAGE-3D report already tested a limited synthetic-to-real classification setting. Knowledge-guided 3D augmentation did not improve the real-image classifier in that pilot (macro-F1 change vs real-only at 1 / 4 / 16 real images per class: −0.025, −0.033, −0.003).

The previous experiment was not a synthetic-only trained ViT. It used a frozen DINOv2 encoder with a linear probe and synthetic augmentation of real training examples. Nevertheless, it is direct evidence that more synthetic imagery does not automatically improve performance.

The new protocol therefore needs to test how synthetic images are generated, not merely how many are generated. In particular, it must address limited underlying asset diversity, appearance realism, backgrounds, and disease-specific visual fidelity.

The current PDF reports 820,502 SAGE image rows across 635 classes with source-grounded KB text. Treat these as the documented candidate coverage, not 820,502 independently usable images or 635 automatically synthesizable classes. The full generation coverage must be audited.

## 4. Recommended experimental scales

Planning calculator example (proposed targets, not claimed available assets or promised results): 100 classes × 10 independent 3D assets per class × 8 camera views per asset × 4 rendering conditions per view = 32,000 synthetic images from 1,000 independent parent assets. Counts are arithmetic planning targets before quality-control exclusions, not verified achievable output. Different views of the same parent asset are strongly correlated.

Start with 20–50 carefully selected classes, then expand to 100, 250, and ultimately the maximum supported class set. Coverage should increase only after real-domain results justify the next stage.

Training on rendered versions of real SAGE images is not strictly synthetic-only in information provenance. The pixels may be synthetic, but the reconstructed geometry and appearance are conditioned on real photographs. The study needs a provenance-aware split and a separate procedural-asset experiment if you want a stronger claim of synthetic-only training.

## 5. Master research meta-prompt

### ROLE AND OBJECTIVE

Act as an autonomous research team consisting of: (1) senior agricultural computer vision researcher; (2) Vision Transformer and transfer-learning specialist; (3) synthetic data generation and 3D graphics researcher; (4) domain generalization and synthetic-to-real transfer researcher; (5) plant disease expert responsible for biological label fidelity; (6) large-scale dataset engineer; (7) PyTorch and HPC/Slurm engineer; (8) machine learning evaluation and statistics researcher; (9) scientific visualization and publication engineer.

Design and implement a complete experimental research program named **DiseaseNet: Knowledge-Grounded Synthetic-to-Real Learning for Scalable Plant Disease Recognition.**

The central task is to train a simple visual disease classifier predominantly or entirely on synthetic disease images generated using the SAGE-Extend 3D protocol, and evaluate classification on independently held-out real SAGE disease images. The ambition is to support hundreds of crop–disease combinations.

This study must be technically simple at the classifier level and rigorous at the data and evaluation level. The primary research contribution is NOT a complicated classification head. It is determining whether large-scale, source-grounded synthetic images can provide transferable disease supervision across crops, conditions, viewpoints, and image domains.

Do not assume high accuracy. Investigate which design choices maximize independently measured real-image performance without contaminating the test set.

---

### PART A — RESEARCH FOUNDATIONS

**A1. Read the complete existing project.** Locate and read: the latest 41-slide SAGE_3D advisor update; the SAGE-Extend 3D codebase; all dataset manifests; SAGE knowledge-base metadata; structured symptom specifications; SAM 3D reconstruction code; pose-alignment implementation; geometry editing operators; GaussianEditor appearance pipeline; two-sided leaf representation; disease-stage handling; evidence-grounding rules; render-to-real gap analysis; previous minimal downstream classification experiment; existing DINOv2 evaluation code; previous failure reports; Slurm and GPU execution scripts.

Preserve all existing results. Create a separate `DiseaseNet/` research directory or an equivalent well-isolated project structure. Do not rewrite SAGE-Extend. Reuse its validated capabilities through stable interfaces.

Existing experimental context — the previous pilot established: 15 reconstructed SAGE specimens; successful input-view pose refinement, but not independently verified 3D reconstruction; KB-controlled deformation operators; volumetric gall generation; surface-specific symptom representation; geometry-aware and appearance-aware editing; multi-view rendering; a source-grounded edit provenance system; negative results for minimal synthetic augmentation. The new experiment must build on these results. The previous negative augmentation result is the starting scientific challenge, not something to hide.

Create: `reports/00_existing_evidence_and_research_gaps.md`

**A2. Study InsectNet and WheatNet.** Read the actual papers, not only search summaries:
- InsectNet: Real-time identification of insects using an end-to-end machine learning pipeline — https://doi.org/10.1093/pnasnexus/pgae575
- WheatNet: A Lightweight Convolutional Neural Network for High-throughput Image-based Wheat Head Detection and Counting — https://arxiv.org/abs/2103.09408

From InsectNet extract: self-supervised visual representation learning; global-to-local fine-tuning; large-class recognition; long-tail class handling; fine-grained recognition; domain transfer; uncertainty estimation; OOD detection; conformal prediction; accuracy versus class-frequency analysis; deployment-oriented efficiency.

From WheatNet extract: lightweight backbone design; multi-scale visual feature aggregation; computational efficiency; robustness to scale variation; agricultural field imagery. Do not claim WheatNet performs disease classification. Do not copy its counting/detection objective into a classification task without justification.

Develop DiseaseNet as an independent project. Document which ideas were directly adopted, which were modified, and which were rejected. Create: `reports/01_literature_and_method_design.md`

---

### PART B — RESEARCH QUESTIONS AND HYPOTHESES

**B1. Primary hypothesis.** A pretrained visual classifier fine-tuned using diverse, biologically grounded synthetic plant disease images can learn transferable crop–disease features that generalize to independently held-out real images. Test this rather than assume it.

**B2. Secondary hypotheses.**
- H1 — Synthetic IID scaling: increasing the number of independent synthetic assets and the variety of rendered conditions improves performance on unseen synthetic parent assets.
- H2 — Synthetic-to-real transfer: increasing synthetic diversity improves real SAGE classification, provided the images retain genuine disease-discriminative features.
- H3 — KB contribution: structured disease-specific synthetic editing improves real-domain performance over generic synthetic appearance editing.
- H4 — Geometry contribution: knowledge-guided 3D geometric variation adds transferable features beyond 2D synthetic editing and appearance-only 3D generation.
- H5 — Domain randomization: variation in camera, background, lighting, scale, texture and occlusion reduces synthetic-to-real domain gap.
- H6 — Backbone choice: a pretrained ViT fine-tuned on synthetic data generalizes better than a randomly initialized ViT under realistic compute budgets.
- H7 — Class scaling: as the number of disease classes increases, real-domain macro-F1 and balanced accuracy follow a measurable scaling relationship.
- H8 — Long-tail recognition: class-balanced synthetic generation can improve performance on underrepresented disease classes relative to imbalanced generation.
- H9 — Hybrid adaptation: a small, strictly separated set of real adaptation images can improve an existing synthetic-pretrained model. This must be reported separately from synthetic-only training.
- H10 — Trustworthy prediction: confidence calibration, rejection, and conformal prediction can improve reliability when the classifier encounters unfamiliar classes or domain shifts.

---

### PART C — DATASET AUDIT AND CLASS ONTOLOGY

**C1. Audit SAGE class coverage.** The existing presentation reports 820,502 image rows and 635 candidate crop–disease classes with source-grounded KB descriptions. Do not assume all rows are valid, unique, available, or appropriate for evaluation. Verify: total crop count; total disease labels; unique crop–disease class count; valid image paths; downloaded image availability; duplicates; per-class sample counts; crop naming consistency; disease synonyms; healthy classes; ambiguous disease labels; multiple disease annotations; dataset-source identifiers; image licenses; KB source availability; supported geometric operators; supported appearance attributes.

Create a deterministic canonical class identifier `crop_id::disease_id` (e.g. `tomato::leaf_mold`). Do not collapse visually similar diseases into one class without a documented taxonomy decision. Do not create a new class for a mere change in severity or camera viewpoint.

Class mapping fields: `class_id`, `crop_name`, `disease_name`, `pathogen_type`, `affected_organs`, `kb_source`, `symptom_stage`, `source_dataset`, `real_image_count`, `synthetic_generation_eligible`, `generation_method`, `exclusion_reason`.

Save: `metadata/class_ontology.parquet`, `metadata/class_coverage.csv`, `reports/02_sage_dataset_audit.md`

**C2. Define class eligibility.**
- Tier A — Fully supported: usable input assets, source-grounded symptoms, valid target masks, editing capabilities, and sufficiently clear disease labels.
- Tier B — Appearance-supported: disease manifestations are principally color or texture changes, and the appearance editor has an appropriate evidence-grounded instruction.
- Tier C — Geometry-supported: disease symptoms include curvature, puckering, gall formation, deformity, or other supported structural attributes.
- Tier D — Unsupported: no reliable input, insufficient biological evidence, unsupported transformation, contradictory annotation, or repeated quality-control failure.

Never silently exclude unsupported classes. Report total coverage at every tier. Maximize coverage without pretending that every disease has a meaningful 3D geometry operation. For Tier B, retain appearance-only generation. For Tier C, use geometry plus appearance where supported. For Tier D, attempt an approved alternative generation pathway or mark the class unsupported. Do not invent symptoms to fill a class quota.

---

### PART D — DATA LEAKAGE PREVENTION

**D1. Establish the real evaluation set first.** Before generating any images, create a locked real SAGE evaluation manifest. Group images by: original source identifier; specimen identifier if available; near-duplicate cluster; image hash; perceptual similarity cluster; dataset provenance; acquisition session when known. Because SAGE may not contain reliable specimen identifiers, treat same-specimen grouping as unresolved where evidence is missing and apply conservative duplicate checks. Do not claim that pHash-disjoint images are guaranteed specimen-independent. Split by group when possible.

Real-domain partitions: (1) real adaptation training pool, optional; (2) real validation pool, permitted for model development under an explicitly declared real-assisted setting; (3) real final test set, completely locked; (4) real open-set unknown-class test pool.

No final-test image or derived asset may be used for: 3D reconstruction training input; editing prompts; render generation; background extraction; synthetic training; hyperparameter selection; domain-gap correction; pseudo-labeling; preprocessing parameter estimation tied to the test set; model selection.

Maintain a separate purely synthetic validation configuration to support a strict no-real-label selection result.

Provenance restriction: any 3D reconstruction originating from a real SAGE image is connected to that real source. This connection must persist through every synthetic render and editing variation. Create a provenance graph `source image → segmentation → 3D asset → edit → rendered image`. Store root source IDs. Enforce no overlap between training roots and locked real evaluation roots. Make leakage checks executable assertions.

---

### PART E — LARGE-SCALE SYNTHETIC DATASET GENERATION

**E1. Main generator.** Real source image or permitted independent plant asset → organ segmentation → SAM 3D reconstruction when appropriate → pose refinement → anatomical QC → structured KB retrieval → stage-consistent symptom specification → geometry editing when supported → appearance editing → multi-view rendering → compositing/domain randomization → biological QC → dataset manifest. The output should consist of actual image files and metadata, not merely rendering instructions. Use existing 3D assets when possible. Avoid reconstructing the same source repeatedly.

**E2. Independent asset diversity.** The number of independent plant assets is more important than simply multiplying camera views. A dataset with 100,000 images rendered from ten plants is not equivalent to 100,000 independent plant observations. Vary: plant identity; organ geometry; crop cultivar where known; shape; age or phenological stage when supported; disease presentation; symptom extent; organ orientation; source acquisition conditions; anatomical background. Use distinct source images and assets wherever available. Do not treat transformations of the same parent object as independent specimens. Track effective independent asset counts per class.

**E3. Disease-stage generation.** For each eligible crop–disease class, construct supported stage variants (e.g. early, intermediate, advanced). Only generate stages described by reliable KB evidence. Do not arbitrarily apply the same three stages to all diseases. Define per stage: visible symptoms; color patterns; geometric effects; affected organs; lesion distribution; severity constraints; prohibited combinations. Stage variations must remain within one class label. Maintain a `stage_id` for analysis.

**E4. Geometry variation.** Use supported operators from SAGE-3D: puckering; local distortion; thickening; signed curling where direction is grounded; gall outgrowth; anatomically constrained volumetric deformation; surface-specific appearance; local depressions where supported. Vary parameters within plausible ranges. Do not generate unrealistic extreme cases solely to increase sample counts. Use biological validation and QC to narrow ranges. Keep the unchanged parent asset as one of the generation controls. Do not force geometric deformation for diseases with appearance-only manifestations.

**E5. Camera sampling.** For each accepted asset, generate multiple views (pilot settings 4, 8, 12, 18). Avoid uniformly using extreme camera orientations. Include realistic photographic angles. Measure and report: organ visibility; target symptom visibility; occlusion; image foreground coverage; camera elevation; camera azimuth; projection scale. Reject views where the intended disease evidence is not visible if the label is defined by visible symptoms, or label those views as ambiguous and evaluate them in a separate study.

**E6. Background and appearance randomization.** Rendering domains: R0 plain controlled background; R1 realistic agricultural image background, with permission and provenance; R2 background randomization; R3 illumination variation; R4 camera style variation; R5 combined environmental variation; R6 optional evidence-preserving appearance harmonization. Include: shadow variation; exposure; white balance; background texture; camera noise; resolution changes; blur; mild occlusion; organ scale; different field-of-view crops. Ensure domain randomization does not erase the disease symptom or create impossible crop morphology. Avoid simply replacing backgrounds with arbitrary photographs that make plants look composited.

**E7. Synthetic generation controls.** Generate paired variants for the same parent asset: (1) original 3D render; (2) generic appearance edit; (3) KB appearance-only edit; (4) KB geometry-only edit; (5) KB geometry-plus-appearance edit; (6) knowledge-grounded 2D synthetic edit; (7) domain-randomized KB 3D render; (8) domain-randomized KB 2D render. Maintain identical class coverage and synthetic counts across comparisons. Do not let the proposed arm have more independent parent plants than the baselines. This is essential to isolate the effect of KB and 3D.

---

### PART F — GENERATION QUALITY CONTROL

**F1. Technical QC.** Automatically reject: invalid rendered images; blank outputs; missing target organs; severe anatomy collapse; invalid splat parameters; extreme deformation artifacts; clipped plant regions; unusable resolution; severe texture noise; obvious compositing defects; nonvisible target symptoms; duplicate renders; empty or incorrect class labels.

**F2. Biological QC.** Audit: correct crop identity; correct disease label; supported symptoms; correct affected organ; disease-stage consistency; anatomical plausibility; lesion and deformation localization; preservation of unaffected regions; unsupported symptom generation. Do not use a VLM's score as the sole biological acceptance rule. Use a small independently reviewed sample to calibrate filters. Document false rejections and false acceptances.

**F3. QC ablation.** Compare: no filtering; technical filtering only; technical plus KB-consistency filtering; technical plus independent biological review/calibrated filtering. Measure how the filters affect downstream real-domain accuracy. Avoid introducing label leakage through QC models trained on the locked real test set.

---

### PART G — SCALE THE SYNTHETIC DATASET

**G1. Stagewise scaling protocol.**
- Stage 0 — Debugging: 20 classes; 3 independent assets per class if available; 4 viewpoints; 2 rendering conditions. Goal: verify the complete pipeline.
- Stage 1 — Controlled pilot: 50 classes; at least 5 independent assets per class where available; 8 views; 4 rendering conditions. Goal: establish synthetic IID and real-domain transfer baselines.
- Stage 2 — Broad experiment: 100 classes; target 10 independent assets per class; 8–12 views; 4 rendering conditions. Goal: study class scaling, domain variation and KB benefits.
- Stage 3 — Larger benchmark: 250 classes where generation and real test support exist; target 10–20 independent assets per class; 8–12 views; 4–8 rendering conditions. Goal: long-tail and broad crop-disease classification.
- Stage 4 — Maximum coverage: extend to the largest audited eligible class set, potentially up to the currently documented 635 candidate KB classes. Do not launch automatically until the earlier scales establish reasonable transfer.

Calculate GPU, storage, and annotation costs before each stage. Do not impose numerical quotas that require inventing fake independent plant assets.

---

### PART H — DISEASENET CLASSIFIER

**H1. Core classifier architecture.** Use a simple pretrained image encoder followed by a classification head. Primary candidate: ViT-S/16 or an equivalently sized pretrained ViT, using an available checkpoint with documented pretrained data and license. Input RGB image → ViT encoder → pooled visual representation → linear projection → crop-disease logits → softmax probabilities. The model must perform image-only inference. Do not require the disease KB or the 3D generator at inference in the main experiment; the generation engine is a training-time component.

**H2. Classifier candidates.** M1 ResNet-50 (conventional baseline); M2 ViT-S/16 (primary simple ViT); M3 DINOv2 ViT-S/14 (self-supervised pretrained representation); M4 DINOv2 ViT-B/14 (larger pretrained baseline if feasible); M5 optional WheatNet-inspired lightweight multi-scale classifier. Keep M2 or M3 as the main DiseaseNet classifier. Do not create an unnecessarily complex architecture before testing these baselines.

**H3. Training modes.** For each main encoder compare: T0 frozen encoder + linear classifier; T1 last transformer blocks unfrozen; T2 LoRA or parameter-efficient adaptation if supported; T3 full encoder fine-tuning; T4 randomly initialized model trained on synthetic only, as a control. Use the same train/validation partitions for all modes. Document generic pretrained data exposure. Do not call a pretrained ViT entirely synthetic-trained if its initialization learned from real photographs; use the phrase **synthetic-supervised fine-tuning of a pretrained visual encoder**. A strict synthetic-from-scratch setting must be separately labeled.

**H4. Recommended training defaults** (starting points, not known optima; tune on an approved validation split; never tune against final real test outcomes): input 224×224; batch size 32–128 effective, determined by measured GPU memory; AdamW; cross-entropy; linear head LR 1e-3 initial search point; full-backbone LR 1e-5 to 5e-5; weight decay 0.05; warmup 5% of steps; cosine decay; 20–50 epochs; mixed precision only when supported by hardware; gradient accumulation; label smoothing off vs modest; class balancing uniform vs inverse-frequency vs effective-number sampling; 3 seeds for shortlisted configurations.

---

### PART I — IID AND OUT-OF-DOMAIN EXPERIMENTS

**I1. Synthetic IID.** Train on synthetic images from selected parent assets; validate on different synthetic parent assets; test on a third disjoint parent-asset set. The same plant asset must never appear in multiple partitions, regardless of camera or rendering condition. Evaluate accuracy, macro-F1, balanced accuracy, per-class recall, top-5 accuracy, confusion, calibration. Also run a deliberately view-disjoint diagnostic, clearly labeled as a weaker generalization test if the parent assets overlap. Main IID claims must use independent parent assets.

**I2. Real SAGE domain-shift evaluation.** Train on synthetic images only; select checkpoints using synthetic validation (strict version); evaluate on the locked real SAGE test set restricted to the same crop–disease classes. This is the primary DiseaseNet benchmark. Report overall accuracy, macro-F1, balanced accuracy, top-5, per-class recall, per-crop accuracy, performance by disease family, by class frequency, by anatomical organ, by geometric versus appearance-dominant disease, confidence calibration, abstention performance. Compare synthetic IID and real-domain performance on the exact same class ontology and compute the generalization gap.

**I3. Open-set OOD.** Reserve disease labels completely absent from classifier supervision. Evaluate unfamiliar-class detection separately with maximum softmax probability, energy score, optional feature-distance score, optional conformal prediction. Report AUROC (known vs unknown), FPR at 95% TPR when appropriate, AUPR, selective prediction accuracy, coverage versus risk. Do not report unseen-class top-1 accuracy as though the closed-set classifier could output labels absent from its head.

**I4. Cross-crop transfer (optional).** For shared disease categories or pathogen groups across crops, only where the biological mapping is defensible, separate: seen crop / seen disease; unseen crop / related disease phenotype; seen crop / unseen disease; unseen crop / unseen disease. For unseen labels, evaluate representation transfer or open-set behavior rather than closed-set classification.

---

### PART J — SYNTHETIC DATA ABLATIONS

**J1. Dataset size.** Synthetic image budgets per class: 25, 50, 100, 250, 500, 1,000 (larger where feasible). Distinguish: more images from existing assets; more independent assets; more disease-stage variation; more domain randomization. Do not collapse these four sources of diversity into one variable.

**J2. Independent plant diversity.** Fix total synthetic image count; vary independent asset count (e.g. 2×100, 5×40, 10×20, 20×10 renders). These require sufficient real independent assets. This may be the most important scaling ablation.

**J3. KB contribution.** Compare generic synthetic images; source-grounded appearance editing; source-grounded geometry editing; joint KB geometry plus appearance; mismatched-KB negative control; wrong-stage KB negative control. Use equal parent assets and render budgets.

**J4. Realism and domain randomization.** Compare plain renders; realistic background composition; background randomization; illumination variation; camera simulation; appearance harmonization; combined domain randomization. Run each on the same set of parent assets.

**J5. 2D versus 3D.** Compare synthetic 2D image editing; unedited 3D renders; generic 3D editing; KB-guided 3D appearance-only; KB-guided 3D geometry-only; full KB-guided 3D editing. A gain must be measured on real SAGE test images. Do not use multi-view consistency alone to claim a classifier benefit.

---

### PART K — DOMAIN-GAP ANALYSIS

**K1. Feature distributions.** Extract features with a common frozen pretrained encoder for real SAGE images, unedited 3D renders, generic edited renders, KB-guided edited renders, domain-randomized renders. Visualize with PCA and UMAP for qualitative inspection, plus numerically meaningful domain-distance measurements. Do not claim domain alignment from UMAP proximity alone.

**K2. Background shortcut detection.** Evaluate original image; background removed; background swapped; foreground-only; symptom-region-only; background-only negative control. If a model predicts disease reliably from backgrounds alone, investigate leakage or class-correlated compositing.

**K3. Symptom grounding.** Use saliency or attention analysis to check whether the classifier attends to actual symptoms; compare with independently annotated symptom regions where available. Do not treat raw attention weights alone as a faithful explanation.

**K4. Failure taxonomy.** Analyze real test errors by: wrong crop; wrong disease, correct crop; appearance similarity; synthetic texture artifact; geometric mismatch; viewpoint shift; background shift; occlusion; disease-stage mismatch; low-resolution symptoms; label ambiguity.

---

### PART L — GLOBAL-TO-LOCAL TRANSFER (InsectNet-inspired, clearly separate)

- **L1 Strict synthetic-only (primary):** pretrained generic encoder → synthetic disease fine-tuning → real test. No real labeled disease images used for model selection or supervised adaptation.
- **L2 Few-shot real adaptation:** synthetic fine-tuned encoder → adapt on 1, 4, 8, 16, 32 real images/class, with disjoint adaptation, validation, and final test images; compare with identical few-shot real training from the generic pretrained encoder (does synthetic training give a better initialization?).
- **L3 Real-only benchmark:** same pretrained model fine-tuned on available real training images — an upper reference, not a fair synthetic-only method.
- **L4 Mixed real plus synthetic:** 100/0, 75/25, 50/50, 25/75, 0/100 real/synthetic; match total training steps; distinguish fixed-real-data from fixed-total-data experiments. Secondary to the synthetic-only claim.

---

### PART M — CLASS SCALING AND LONG-TAIL EVALUATION

Run increasing class coverage: 20, 50, 100, 250, maximum audited eligible classes. Use nested class subsets wherever possible. Do not repeatedly choose easier classes as the experiment scales. Select cohorts by crop diversity, symptom modality, class frequency, biological ambiguity, generation feasibility, taxonomic variety. Report results for common subsets as well as the growing full set. For every stage report: number of classes, crops, parent assets per class, synthetic images, real test images, accuracy, macro-F1, balanced accuracy, GPU-hours, failed generation rate, excluded class count.

---

### PART N — STATISTICAL DESIGN

Use specimens or source groups, not rendered views, as the independent unit. For the real SAGE test set: group-aware resampling CIs; paired classifier comparisons; class-level effect distributions; per-crop results; multiple seeds; correction for primary multiple comparisons; primary metric defined before final tests; failures reported, not only successful classes.

Primary endpoint: **real SAGE macro-F1 on the closed-set synthetic-trained benchmark.** Secondary: balanced accuracy; top-1; top-5; synthetic IID macro-F1; IID-to-real gap; worst-quartile class recall; OOD AUROC; calibration error; accuracy per unit of compute. Include a power or precision assessment based on pilot variance. Do not invent confidence intervals or expected accuracy.

---

### PART O — COMPUTE PLANNING AND GPU EXECUTION

Inspect hardware: GPU model; VRAM; CUDA; PyTorch compatibility; BF16/FP16 support; existing checkpoints; storage; CPU memory; Slurm policy; queue status. The existing SAGE-3D project used two 12 GB TITAN X GPUs; do not assume an A100 configuration fits.

Implement gradient accumulation; mixed precision only when safe; gradient checkpointing when useful; persistent dataset cache; streaming image loading; resume support; checkpointing; early stopping; multi-GPU training if supported. Record GPU-hours separately for reconstruction, editing, rendering, QC, classifier training, evaluation.

Mandatory measured compute estimates for 20-, 50-, 100-, 250-class and maximum coverage. Distinguish new-asset reconstruction cost from cheap rendering of an existing asset. Do not multiply full GaussianEditor optimization cost by every rendered image.

---

### PART P — EXPERIMENTAL EXECUTION ORDER

- Phase 0 Audit: SAGE dataset audit; class eligibility mapping; existing assets inventory; leakage-safe splits; hardware analysis.
- Phase 1 Minimal classifier feasibility: 20 eligible classes; generate or reuse sufficient independent assets; small balanced dataset; train frozen pretrained ViT + linear head and fine-tuned pretrained ViT; evaluate synthetic IID and real SAGE. Do not proceed blindly if synthetic IID fails; if IID is high but real is poor, prioritize domain-gap experiments.
- Phase 2 Generator comparison: 2D KB editing; unedited 3D; KB 3D; KB 3D + domain randomization. Select using synthetic validation and a clearly designated real development benchmark if permitted; keep a strict no-real-label-selection result separately.
- Phase 3 50–100 classes: ViT and DINOv2 fine-tuning; asset-diversity scaling; image budgets; domain randomization; real SAGE performance.
- Phase 4 250-class expansion only if earlier phases establish meaningful generalization and coverage.
- Phase 5 Maximum audited coverage; report unsupported classes.
- Phase 6 Few-shot real adaptation (global-to-local), separately.
- Phase 7 OOD and uncertainty: unknown-disease detection, calibration, selective prediction.

---

### PART Q — SCIENTIFIC FIGURES (from executed results only)

1 DiseaseNet pipeline · 2 synthetic dataset diversity · 3 class coverage · 4 synthetic vs real (paired by class, not specimen) · 5 IID and real-domain learning curves · 6 backbone comparison (ResNet, ViT-S, DINOv2-S, optional DINOv2-B) · 7 synthetic-data scaling (macro-F1 vs images, independent assets, classes) · 8 domain-gap analysis with quantitative context · 9 generation ablations (KB, geometry, 2D, 3D, randomization, filtering) · 10 real SAGE confusion analysis · 11 long-tail accuracy · 12 global-to-local adaptation · 13 OOD and calibration · 14 successes and failures with labels, predictions, confidence and symptom-level interpretation. Publication-quality styling; vector exports where feasible. Never fill missing plots with imagined numbers.

---

### PART R — CODEBASE STRUCTURE

```
DiseaseNet/
├── configs/{datasets,generation,classifiers,training,evaluation}/
├── data/{manifests,synthetic,real_splits,metadata}/
├── src/{generation,data_quality,provenance,datasets,models,training,evaluation,domain_gap,visualization}/
├── experiments/{E0_feasibility,E1_synthetic_iid,E2_synthetic_to_real,E3_kb_geometry,E4_domain_randomization,E5_scale_classes,E6_global_to_local,E7_ood}/
├── scripts/{audit_classes,make_splits,build_synthetic_dataset,validate_synthetic_dataset,train_vit,train_dinov2,evaluate_iid,evaluate_real_sage,evaluate_ood,analyze_domain_gap,generate_all_figures}.py
├── slurm/  checkpoints/  results/  figures/  reports/
```
Adapt to actual source code and installed libraries. Provide fully executable scripts and configuration examples.

---

### PART S — QUALITY ASSURANCE AND REPRODUCIBILITY

Automated tests for: class ontology integrity; duplicate detection; no evaluation leakage; parent-asset split integrity; KB source tracing; disease-stage validation; correct synthetic labels; image file integrity; dataset balance; batch tensor shapes; model forward pass; checkpoint loading; deterministic evaluation; metric correctness; resume functionality; cross-stage artifact consistency.

For every run save: git commit; environment; training config; dataset manifest version; parent asset IDs; generator version; seed; training logs; checkpoint; per-image predictions; class mapping; metrics; GPU runtime.

---

### PART T — FINAL DELIVERABLES

1 `DISEASENET_MASTER_PROTOCOL.md` · 2 `SAGE_CLASS_AUDIT.md` · 3 `CLASS_COVERAGE.csv` · 4 `SYNTHETIC_DATASET_SPECIFICATION.md` · 5 `IID_OOD_SPLIT_PROTOCOL.md` · 6 `MODEL_ARCHITECTURE.md` · 7 `TRAINING_CONFIGURATIONS.md` · 8 `DOMAIN_GAP_EXPERIMENTS.md` · 9 `EXPERIMENT_MATRIX.csv` · 10 `COMPUTE_ESTIMATE.md` · 11 `PILOT_RESULTS.md` · 12 `MAIN_RESULTS.md` · 13 `STATISTICAL_REPORT.md` · 14 `FAILURE_ANALYSIS.md` · 15 `ADVISOR_PRESENTATION.pdf` · 16 `RESEARCH_PROGRESS.md`. Include executable code, actual result figures, and tables with measured values only.

The final report must answer:
1. How many SAGE classes can be represented synthetically?
2. How many independent assets are available per class?
3. How many images can be generated within the GPU budget?
4. How much synthetic data is needed for stable IID classification?
5. What happens when training only on synthetic data and testing on real SAGE images?
6. How much does real-domain performance change with class count?
7. Does adding more views help, or do more independent assets help more?
8. Does source-grounded disease editing outperform generic rendering?
9. Does 3D geometry improve classification beyond synthetic 2D?
10. Which backgrounds and textures harm real-domain transfer?
11. Does pretrained ViT outperform training from scratch?
12. Does synthetic pretraining help low-shot real adaptation?
13. Which disease categories transfer poorly?
14. How trustworthy are the predictions on unfamiliar diseases?
15. Is the method feasible at hundreds of classes?

### FINAL EXECUTION INSTRUCTIONS (for the executing agent)

Not only a proposal: (1) inspect the codebase; (2) audit SAGE and available 3D assets; (3) implement a leakage-safe data pipeline; (4) define the feasible class set; (5) build the synthetic generation manifest; (6) produce a small real synthetic dataset; (7) train an actual simple ViT; (8) evaluate synthetic IID; (9) evaluate real SAGE domain shift; (10) diagnose initial errors; (11) run controlled small-scale generator ablations; (12) estimate compute for larger coverage; (13) produce a reproducible protocol; (14) generate scientific plots; (15) save exact continuation instructions.

First complete a 20-class pilot before spending large amounts of GPU time. Use a pretrained ViT as the primary classifier. Strict real-test isolation. Do not tune on the final real test set. Do not manufacture higher accuracy by removing difficult classes after viewing test performance. Do not call images from the same parent 3D asset independent specimens. Do not claim synthetic-only training if the pretrained backbone or source-asset provenance requires a qualification. Do not promise a particular accuracy.

**Ultimate goal: build the largest scientifically defensible knowledge-grounded synthetic plant disease dataset possible, and determine how well a simple, synthetic-supervised ViT can classify real SAGE disease images across many crop–disease categories.**

## 6. The first experiment to actually run

A 20-class, 2,000–10,000 synthetic-image pilot to diagnose transfer before building hundreds of thousands of images.

| Component | Initial choice |
| --- | --- |
| Classifier | Pretrained DINOv2 ViT-S/14 |
| Classification head | Single linear layer |
| Training | Synthetic images only |
| IID test | Synthetic images from unseen parent assets |
| Real test | Held-out SAGE photos of identical classes |
| Primary metric | Real-domain macro-F1 |
| Secondary metric | Top-1 accuracy |
| First ablation | KB 3D versus unedited 3D versus KB 2D |
| Second ablation | Number of independent plant assets |
| Third ablation | Background and illumination randomization |

Do not judge the whole concept by one frozen linear probe: compare it with last-block and full-backbone fine-tuning. The largest potential mistake would be creating 500,000 renders from a few dozen reconstructed plants. Make the first principal figure real SAGE macro-F1 versus number of independent synthetic parent plants, alongside a separate curve for number of rendered images.

Overarching paper narrative: does source-grounded synthetic supervision scale to real-world fine-grained agricultural disease recognition, and which components make that transfer possible?
