# DiseaseNet Stage 1 master prompt (received 2026-10-16; saved for traceability)

Note: the brief names `SAGE_3D (2).pdf`. That file is not on this machine; `~/Downloads/SAGE_3D.pdf` is the
30-page deck of 2026-10-12. The latest deck (43 pages, incl. DiseaseNet Stage 0 slides) is the Overleaf project
6ac80070 compiled locally; it is the version used here.

Sections (condensed headings; the full text was pasted in the session):
1. Reproduce and audit Stage 0 -> reports/DN01_stage0_audit.md. Treat the Stage-0 locked test as exploratory;
   establish a separate final evaluation protocol.
2. Independent synthetic asset diversity audit (per class: parents, assets, edits, views, lighting, backgrounds,
   stages, rejects, embedding spread, near-duplicate rate); plot images vs independent parents.
3. Synthetic-to-real gap diagnosis: A background shift (+ background-only shortcut), B viewpoint shift
   (input view / narrow / 4 / 8 / 18 views), C disease-feature preservation, D appearance harmonisation.
4. Asset-diversity scaling at fixed image count (few parents x many renders ... many parents x few renders);
   no fake parents from rotations/re-deformations. PRIMARY experiment.
5. Geometry-sensitive vs appearance-sensitive cohorts; generators: GEN2D, KB2D, U3D, KB-appearance-3D,
   KB-geometry-3D, full KB3D, KB2D+KB3D mix; modality x generator interaction.
6. ViT fine-tuning study (keep T0/T3): T1 linear, T2 last 1-2 blocks, T3 LoRA, T4 full, T5 larger ViT later;
   HP search on non-final validation; LLRD, early stopping, WD, small LRs, label smoothing, balanced sampling,
   augmentation, clipping; 3 seeds for selected configs; train/val loss curves; embedding drift.
7. Synthetic IID with parent-level train/val/test partitions; flag classes with too few parents.
8. Real evaluation: Strict S0 (synthetic-only selection) and Adapted S1 (0/1/4/8/16 real shots/class, separate
   real validation; synthetic- vs generic-pretrained with identical shots and budget).
9. KB2D/KB3D mixing at fixed total (100/75/50/25/0) + geometry-only-class mixture; no selection on final test.
10. Class coverage: canonical eligible-class manifest; nested 20 / 50 / 100 / 250 / max; per-class fields.
11. Large-scale classifier: DINOv2 ViT-S/14 primary; ViT-S, ResNet-50, ViT-B/14 secondary; scaling curves.
12. Open-set: held-out classes; energy, MSP, AUROC, FPR@TPR, risk-coverage.
13. Statistics: primary contrasts (KB3D vs U3D; KB3D vs KB2D; KB2D+KB3D vs KB2D; last-block FT vs frozen probe);
    per-seed, paired group bootstrap, crop/class breakdowns, multiplicity correction; recomputed from predictions.
14. Figures 1-16 (data-driven, traceable).
15. Reports: DISEASENET_STAGE1_MASTER_PROTOCOL.md, STAGE0_VERIFICATION.md, DATA_PROVENANCE_AUDIT.md,
    CLASS_ELIGIBILITY.csv, SYNTHETIC_DIVERSITY_ANALYSIS.md, DOMAIN_GAP_DIAGNOSIS.md,
    CLASSIFIER_FINE_TUNING_STUDY.md, GENERATOR_ABLATION_RESULTS.md, IID_REAL_OOD_RESULTS.md,
    SCALEUP_COMPUTE_ESTIMATE.md, FAILURE_ANALYSIS.md, DISEASENET_ADVISOR_UPDATE.pdf, RESEARCH_PROGRESS.md.
    Each separates measured / pilot / inference / hypothesis / planned / blocker.
16. Order: audit -> provenance -> fine-tuning -> IID -> gap diagnosis -> bg/view/appearance ablations ->
    parent-diversity scaling -> 2D/3D mixing -> geometry cohort -> costs -> freeze protocol -> 50 then 100
    classes behind feasibility gates -> advisor deck. No hundreds of GPU-hours without evidence; no tuning on the
    final test; keep negative results; no claims beyond evidence.
Ultimate question: can source-grounded synthetic generation give a scalable training distribution whose
discriminative features transfer to real crop-disease recognition; if KB-3D does not beat 2D, under which
conditions (if any) is geometry justified?
