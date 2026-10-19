# Scale-up compute estimate (beast0: 2 × GTX TITAN X 12 GB, Maxwell; measured 2026-10-15/16)

## Measured unit costs
| step | cost | source |
|---|---|---|
| SAM 3D reconstruction (stage-wise offload) | **1,131 s/asset** mean (median 1,116, max 1,416); peak 11.6 GB → one asset per GPU, nothing else on that GPU | 45 Stage-0 `run_meta.json` |
| SAM ViT-H parent mask + QC | ~10–30 s/class-candidate incl. reading the parquet shard | `select_parents --stage1` log |
| InstructPix2Pix edit (20 steps, 512 px) | 27–35 s alone; 40–90 s when two jobs share a GPU | `logs/stage1/edits_gpu*.log` |
| render 32 images (8 views × 4 conditions) per asset and arm | ~1–3 s (gsplat) incl. pose/plates | `logs/stage1/render_*.log` |
| image storage | ~25 KB/image (384 px JPEG); 1,920 images/arm/replicate ≈ 48 MB; asset ≈ 8.5 MB | `du` |
| DINOv2 ViT-S/14 features | ~45 img/s incl. JPEG decode (cached once) | Stage-0 T0 step |
| T0 probe (logistic regression, 20 classes, 1,920 images) | seconds; LOPO 3 folds < 1 min | `run_probes.py` |
| T4 full fine-tune, 15 epochs, 1,920 images | 491 s alone (72 img/s, bs 64, 5.6 GB) | Stage-0 T3 |
| few-shot S1 (frozen features, 21 fits × 4 weights) | ~5–10 min CPU per arm | `run_fewshot.py` |

Free disk: 21 GB (98 % full). GPU 1 is the reconstruction GPU (GPU 0 carries ~0.5 GB display memory).

## Stage-1 plan costs
| item | new assets | GPU-hours (wall on beast0) | disk |
|---|---|---|---|
| Stage 0R: fixed edits (60 × 2 prompts × 3 seeds) + cross edits (48) | 0 | ~4 GPU-h (~2 h wall, 2 GPUs) | 0.3 GB |
| Stage 0R renders, 12 arms × 3 replicates + view sets | 0 | < 1 | ~2 GB |
| fine-tuning study (40 LOPO runs + 24 final runs) | 0 | ~6 (3 h wall) | < 0.1 GB |
| parent diversity, **2D** (9 new parents × 20 classes; masks + KB/generic edits) | 0 | ~2.5 | 0.3 GB |
| parent diversity, **3D** (same 180 parents) | 180 | **57 GPU-h** (≈ 2.4 days on one GPU) | 1.6 GB assets + 1 GB renders |
| 50-class cohort, 3 parents/class (30 new classes) | 90 | 28 (+1 2D) | 0.8 GB + 1.5 GB |
| 87-class cohort (maximum eligible), 3 parents/class | 201 | 63 | 1.8 GB + 3 GB |
| 87-class cohort at 12 parents/class | 804 | **253** | 7 GB + 8 GB (exceeds free disk with renders) |

## Gates
- G1 (3D parent diversity, 57 GPU-h): only if Stage 0R shows a 3D arm within the 2D noise floor on DEV or a
  geometry-cohort effect worth testing; requested from the user with this table.
- G2 (50 classes, 28 GPU-h) and G3 (87 classes, 63 GPU-h): after G1's answer; 2D-only versions cost ≤ 3 GPU-h and
  can run first. The brief's 100- and 250-class tiers are not reachable: only 87 SAGE classes meet the evaluation
  rules (`CLASS_ELIGIBILITY.csv`).
- Anything ≥ 100 GPU-h (e.g., 87 classes × 12 parents in 3D) needs Nova and an explicit decision.
