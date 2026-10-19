# GPU scaling estimate (from measured medians; local TITAN X)

Measured medians: SAM 3D reconstruction 1070 s; GaussianEditor 200-step run 2159 s; IP2P 2D edit 57 s; geometry edit (incl. 18-view render) 44 s.

Per specimen, the full arm set is: 1 reconstruction; 2D B×3, C×3, C-NG, C-MM, Bmv×8, Cmv×8, C-stage×3, I×3 = 30 IP2P; 3D E, F, K, H = 4 GE runs; J×3, G×3 = 6 geometry edits.

**≈ 3.2 GPU-h per specimen** (one seed for the GE arms).

| Specimens | GPU-h | wall-clock on 2 GPUs (ideal packing) |
|---|---|---|
| 15 | 49 | 24 h (1.0 d) |
| 50 | 162 | 81 h (3.4 d) |
| 150 | 487 | 244 h (10.1 d) |
| 500 | 1624 | 812 h (33.8 d) |

GE dominates (74 % of per-specimen time) because of IP2P calls inside GE. An A100 would be several times faster, but no A100 timing was measured here, so none is claimed.
