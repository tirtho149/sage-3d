# Stage 0 verification (2026-10-16)

Every number below was **recomputed from saved per-image predictions** (`results/E0/t0_predictions.npz`,
`t3_<arm>_real_s<seed>.npz`) with sklearn `f1_score(average="macro", labels=0..19)` and compared with the stored
JSON. Test labels were asserted equal to `data/real_splits/stage0_test.csv` (4,202 rows) for every file.

## Arm scores (real test, macro-F1)

| arm (label as reported) | T0 recomputed | T0 stored | T3 s0 / s1 / s2 recomputed | T3 mean |
|---|---|---|---|---|
| PHOTO | 0.6618 | 0.6618 | 0.611 / 0.639 / 0.609 | 0.620 |
| KB2D | 0.6720 | 0.6720 | 0.604 / 0.634 / 0.607 | 0.615 |
| GEN2D | 0.6693 | 0.6693 | 0.621 / 0.617 / 0.608 | 0.615 |
| KB3D | 0.5905 | 0.5905 | 0.546 / 0.554 / 0.562 | 0.554 |
| U3D | 0.5331 | 0.5331 | 0.478 / 0.538 / 0.491 | 0.502 |

T3 JSON values match the recomputation to < 1e-9 for all 15 files.

## Contrasts (brief table) — all reproduced exactly

| contrast | T0 brief | T0 recomputed | T3 brief | T3 recomputed |
|---|---|---|---|---|
| KB3D − U3D | +0.057 | +0.057 | +0.052 | +0.052 |
| KB3D − PHOTO | −0.071 | −0.071 | −0.066 | −0.066 |
| KB2D − KB3D | +0.082 | +0.082 | +0.061 | +0.061 |
| KB2D − PHOTO | +0.010 | +0.010 | −0.005 | −0.005 |
| GEN2D − KB2D | −0.003 | −0.003 | +0.000 | +0.000 |

## But the arm labels are wrong

The numbers are verified; the **arm definitions are not** — see `reports/DN01_stage0_audit.md` §1. Because of a
mask bug in `edit2d.py`, KB2D and GEN2D are the real parent photos and Stage-0 KB3D is the SAM 3D asset with the
source photo's texture re-projected (+ KB geometry in 4 classes). Correct reading:

| contrast as reported | correct name | T0 | T3 |
|---|---|---|---|
| KB3D − U3D | TEXGEO3D − U3D (photo texture + geometry vs SAM 3D colours) | +0.057 | +0.052 |
| KB3D − PHOTO | TEXGEO3D − PHOTO | −0.071 | −0.066 |
| KB2D − KB3D | PHOTO′ − TEXGEO3D | +0.082 | +0.061 |
| KB2D − PHOTO | PHOTO′ − PHOTO (seed noise) | +0.010 | −0.005 |
| GEN2D − KB2D | PHOTO″ − PHOTO′ (seed noise) | −0.003 | +0.000 |

(′, ″ = the same photos with different 2D-variant / background draws.)

## Status of the Stage-0 test set

Used for all Stage-0 reporting and for one follow-up decision (adding T3 seeds). Kept as **DEV** — exploratory and
diagnostic. Stage-1 confirmatory numbers come only from the FINAL test (untouched `val` partition), evaluated
once per frozen configuration and logged in `results/stage1/FINAL_ACCESS_LOG.csv`.

Reproduce: the recomputation script is inline in the session log; `scripts/deck_stage0.py` recomputes the
paired contrasts with the group bootstrap (`results/E0/t3_paired.json`).
