# 02 — Dataset Protocol (SAGE-only)

## Tiers

| Tier (master prompt) | Status | Source |
|---|---|---|
| 1. Real multi-view of same specimen | **Not available.** Out of scope by decision (2026-10-12). | — |
| 2. Real 3D / measured traits | **Not available.** Out of scope. | — |
| 3. Public single-view images | **Used** (all experiments) | SAGE, HF `tirtho149/SAGE`, CC-BY-4.0 |

Generated data are never treated as ground truth. Synthetic renders are labelled `synthetic` in every manifest that contains them.

## Provenance and licence

- **SAGE:** Arshad, Roy, et al., arXiv:2605.09768, CC-BY-4.0. Images aggregate public benchmarks (column `source`). The licence of each upstream source applies to its images and is not re-verified here; the per-source licence audit is listed as an open item.
- **Local copy:** `~/Desktop/Research/data/raw/SAGE` (48 shards, 20.5 GB).
- **Verified manifest:** `data/metadata/sage_shard_manifest.parquet` (all non-image columns plus shard name).

## Unit, deduplication, splits (`scripts/make_splits.py`)

- **Unit:** one image after exact-duplicate removal (SHA-1 of the image bytes). SAGE has no specimen or camera IDs and no capture dates.
- **Near-duplicates:** 64-bit pHash; pairs with Hamming ≤ 6 are unioned into clusters. A cluster is never split across partitions.
- **Partitions:** build 45 / opt 25 / val 15 / test 15, assigned per cluster and stratified by the cluster's majority class.
- **Frozen split hash:** `data/metadata/splits.sha256` = `1c81461f…688ece`.

| Crop | Rows in SAGE | Unique (SHA-1) | pHash clusters | build / opt / val / test |
|---|---|---|---|---|
| Citrus | 840 | 838 | 765 | 387 / 211 / 122 / 118 |
| Corn | 62,335 | 43,517 | 25,258 | 19,614 / 10,879 / 6,531 / 6,493 |
| Mango | 11,918 | 11,885 | 9,841 | 5,353 / 2,970 / 1,793 / 1,769 |
| Peach | 22,447 | 12,160 | 4,951 | 5,479 / 3,038 / 1,827 / 1,816 |
| Tomato | 208,439 | 126,591 | 53,942 | 56,993 / 31,654 / 18,983 / 18,961 |
| **Total** | **305,979** | **194,991** | | |

**Finding:** 36% of SAGE rows for these five crops are exact byte-duplicates, and near-duplicate clustering compresses further (Tomato 126,591 → 53,942 clusters; Peach 12,160 → 4,951). An image-level random split of SAGE would place copies of the same photo in train and test. This affects any SAGE-based evaluation, not only this project.

## Pilot specimen manifest

- `data/metadata/specimen_manifest.csv` covers 10 candidate classes, ≤300 images each, with columns `specimen_id, sage_filename, sage_source, shard, sha1, width, height, sat, auto_ok, split, cluster`.
- `experiments/configs/pilot.yaml` lists 15 pilot specimens, **all in the build split** (asserted in code). It includes a replacement log: 9 specimens originally chosen before splitting were replaced because they fell into opt/val/test. One build candidate (Peach 0035) was rejected for a watermark over the organ.
- **View annotations** (`view_annotations.csv` in the master prompt): not applicable, since every specimen has exactly one view (`view = source`). Synthetic rendered views are recorded per asset in `rig.json` (azimuth, elevation, extrinsics). They are never mixed into real-image manifests.

## Leakage rules (enforced)

1. Asset sources (SAM 3D inputs, masks, edits) come from **build only**.
2. KB text is class-level and identical for all splits. It is never fitted to test images.
3. E5 test queries come from **test only**. Reference banks come from build only.
4. Any image whose SHA-1 or pHash cluster appears in the test split is rejected from asset building (`src/evaluation/leakage.py`, pending).

## Open items

- Per-source licence audit of SAGE upstream datasets.
- Expert review of the flagged label-quality issues (Squash Downy Mildew; duplicated Citrus KB texts; see `03_knowledge_protocol.md`).
