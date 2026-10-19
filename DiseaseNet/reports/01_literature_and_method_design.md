# 01 — InsectNet and WheatNet: what DiseaseNet adopts, modifies, rejects

Read from the publishers' pages on 2026-10-15: InsectNet, PNAS Nexus 4(1) pgae575
(https://doi.org/10.1093/pnasnexus/pgae575; preprint arXiv:2306.02507) and WheatNet,
arXiv:2103.09408 (HTML v2). InsectNet's OOD and conformal details are in its
Supplementary Information, which was not accessible here; those points are marked.

## InsectNet (insect species identification)

What the paper reports:
- RegNet/ResNet backbones; staged pretraining: weakly supervised on ~3.6 B Instagram images
  (SWAG), then self-supervised on ~12 M unlabeled iNaturalist insect images, then supervised
  fine-tuning on ~6 M expert-curated labeled images.
- 2,526 species (17 orders), per-species counts from 38 to ~136 k (≈ 3 orders of magnitude
  imbalance); test/val 10 images per species.
- Headline 96.4% top-1, ~94% mean per-species accuracy (the paper elsewhere says "over 96%";
  internally inconsistent); 239 species below 75%.
- Long tail: post-hoc AlphaNet transfers from high- to low-accuracy classes (low-accuracy
  species 79.7% → 87.6%, overall 96.4% → 95.1%).
- Global-to-local: regional model for 54 Midwest pest species (~540 k images) from the
  global model: 98.7% / MPCA 96.1% vs 94.3% / 90.2% from scratch; few-shot k = 50/20/10 on
  external sets reported.
- Abstention via an energy-based OOD score (details in SI); conformal prediction sets with
  a stated ≥ 97.5% coverage target (the Fig. 2 example uses 81%; inconsistent).
- Limitations stated: confusable species, scarce/undiverse data, inconsistent public labels,
  phone-resolution limit, need for expert validation.

## WheatNet (wheat head counting)

- Task: density-map counting + localization of wheat heads (not disease classification).
  Authors: Saeed Khaki, Nima Safaei, Hieu Pham, Lizhi Wang.
- Truncated MobileNetV2 (first four bottleneck stages), multi-scale fusion of the four stage
  outputs (upsampled and concatenated), two parallel heads; 4.04 M parameters.
- GWHD: 3,373 images (10 locations); MAE 3.85 / RMSE 5.19 vs Faster R-CNN 4.25 / 5.71
  (43.2 M params); 0.54 s/image on a CPU.

## Design decisions for DiseaseNet

| Idea | Decision | Reason |
|---|---|---|
| Large self-supervised pretrained backbone | **Adopt** (DINOv2 ViT-S/14 primary, ViT-B/14 secondary) | DINOv2 is self-supervised on generic images; matches InsectNet's representation-first design; fits 12 GB (measured) |
| Staged domain self-supervision (iNat insects) | **Modify / defer** | an analogous step would be SSL on unlabeled SAGE `parent` images; must exclude the locked test; Phase 3+ option |
| Global-to-local fine-tuning | **Adopt as L2**: synthetic-supervised model → few-shot real adaptation on the `adapt` partition | mirrors InsectNet's global → regional step; reported separately from synthetic-only |
| Long-tail evaluation (per-class accuracy vs frequency) | **Adopt** (Figure 11) | SAGE eligible classes span 30–69k images |
| AlphaNet post-hoc long-tail fix | **Defer** | first measure whether the tail even transfers |
| Energy-score abstention, conformal sets | **Adopt** (Phase 7, I3) using MSP + energy, split conformal on a synthetic calibration set (and separately a real `val` set in the real-assisted setting) | exact InsectNet SI settings not read; our thresholds will be set on validation only |
| Lightweight multi-scale backbone | **Adopt as optional baseline M5** (MobileNetV2 truncated + multi-scale fusion → classification head) | WheatNet's efficiency idea, not its counting objective |
| Counting/density objective | **Reject** | DiseaseNet is classification; no justification to count |
| "Accuracy transfers from InsectNet" | **Reject** | different task, data and supervision |

Primary classifier for Stage 0: DINOv2 ViT-S/14 + linear head, trained T0 (frozen) and T3
(full fine-tune, fp32, batch 64 with accumulation), primary endpoint real SAGE macro-F1.
