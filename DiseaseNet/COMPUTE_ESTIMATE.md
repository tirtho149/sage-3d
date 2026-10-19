# Compute estimate (Phase 0, Part O) — measured on beast0, 2026-10-15

Hardware: 2× NVIDIA GTX TITAN X 12 GB (Maxwell sm_52; fp16 slower than fp32, no bf16 →
train in fp32), 12 CPU threads, 125 GB RAM, **25 GB free disk**. No local SLURM; Nova via
Globus is configured.

## Measured unit costs

| Step | Cost | Source |
|---|---|---|
| SAM 3D reconstruction (new parent asset) | 1070 s median (17–26 min), peak 11.6 GB, one GPU at a time | SAGE-Extend reports/08 |
| Pose alignment | 78 s | reports/08 |
| Geometry edit (KB operator) | ~45 s incl. 18 renders | reports/08 |
| GaussianEditor appearance edit | ~2200 s | reports/08 — too slow per image; use per asset only if needed |
| IP2P 2D edit | 57 s (GPU) | reports/08 |
| Render of an existing asset | **7 ms per view** (512 px) | measured here |
| Asset on disk | 28 MB (10 MB if mesh + duplicate splat are dropped) | measured here |
| DINOv2 ViT-S/14, frozen + linear | ~230 img/s, 0.4–0.8 GB | measured here (bs 64/128) |
| DINOv2 ViT-S/14, full fine-tune | ~72 img/s at bs 64, 5.6 GB; bs 128 OOM → grad accumulation | measured here |

Reconstruction dominates; rendering and classifier training are cheap.

## Per stage (estimates from the unit costs; assets per class are targets, availability permitting)

| Stage | Classes | New assets | Recon GPU-h | Render (8 views × 4 cond.) | Images | Disk (assets) | Full FT 30 epochs (1 run) | Where |
|---|---|---|---|---|---|---|---|---|
| 0 | 20 | ~45 new (+15 existing) × geometry/appearance arms | ~14 | < 1 min | ~2k per arm | ~0.6 GB | ~15 min | beast0 |
| 1 | 50 | ~250 | ~75 | < 5 min | ~8k per arm | ~2.5–7 GB | ~1 h | beast0 (3–4 days, disk tight) or Nova |
| 2 | 100 | ~1,000 | ~300 | minutes | ~32k per arm | 10–28 GB | ~4 h | Nova |
| 3 | 250 | 2,500–5,000 | 750–1,500 | < 1 h | 80–160k | 25–140 GB | 10–20 h | Nova (scavenger, requeue) |
| 4 | ≤ 195 eligible today | — | — | — | — | — | — | needs KB/operator coverage first |

Notes
- Stage 4 is bounded by the audit: **195 eligible classes now**, not 635 (617 classes lack KB
  text; 361 have < 30 unique images). Stage 3 (250 classes) is not reachable without
  extending KB coverage or relaxing eligibility.
- "Images per arm": each ablation arm (unedited 3D, KB 3D, KB 2D, ...) needs the same parent
  assets and budget, so total images = arms × the column.
- Edits are per asset, not per image: GaussianEditor cost is never multiplied by views.
- Disk: Stage 1 locally needs pruning (keep splat only) or moving outputs to Nova.
