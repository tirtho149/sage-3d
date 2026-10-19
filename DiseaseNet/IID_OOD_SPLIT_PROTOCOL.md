# IID / domain-shift / open-set split protocol (Phase 0, Part D) — LOCKED 2026-10-15

Built by `scripts/make_splits.py` before any synthetic image exists. Lock:
`data/real_splits/LOCK.json` (test sha1 list hash
`0fb2898e5edec92fa2afe7148ddfe6e4ccb3560edd191c5b04ad8c9de4e684c2`). Every load verifies it
(`src/provenance/leakage.py: Guard`); tests: `tests/test_splits.py` (7 pass).

## Real SAGE partitions (195 eligible classes, 426,189 unique images)

| Partition | Unique images | May be used for |
|---|---|---|
| parent | 155,651 | source photos for 3D reconstruction, 2D edits, backgrounds (generation roots) |
| adapt | 59,005 | few-shot real adaptation only (H9 / L2), reported separately |
| val | 58,092 | real validation, only in the declared real-assisted setting |
| **test** | **116,946** | **locked final test; nothing derived from it, never used for selection** |
| openset | 24,523 | 19 classes reserved entirely as unknown classes (I3) |
| excluded_openset_overlap | 11,972 | known-class images near-duplicate to an open-set image; unused |

Known classes: 176. Open-set classes: 19 (10% per tier, seeded, excluding the 5 pilot
classes); listed in `LOCK.json`.

## Rules

1. **Unit = near-duplicate group.** Exact duplicates share a node; groups join any images
   within Hamming distance ≤ 6 of 64-bit pHash (8 × 8-bit bands guarantee no such pair is
   missed). 235,142 groups; the two largest (7,274 and 3,727) come from transitive chaining
   and are kept merged (conservative); no class has a single group holding > 50% of it.
2. **One partition per group everywhere**, so the same photo can never be train in one class
   and test in another.
3. **Provenance:** every synthetic image stores its root photos; `Guard.assert_generation_roots`
   allows only `parent` roots, and `Guard.assert_not_near_test` rejects any derived image within
   6 bits of a locked test image. The 15 existing SAM 3D assets' source photos are forced into
   `parent` (verified: 15/15).
4. **Synthetic IID** splits are by parent asset (train / val / test parent assets disjoint);
   renders of one asset never cross partitions. A view-disjoint split is only a labelled
   diagnostic.
5. **Strict synthetic-only setting:** checkpoints selected on synthetic validation only.
   The real `val` partition exists only for a separately reported real-assisted setting.
6. **Wording:** a pretrained encoder fine-tuned on these images is "synthetic-supervised
   fine-tuning of a pretrained visual encoder", and assets reconstructed from real `parent`
   photos are "real-derived synthetic" — not "trained on synthetic data only".

## Known limitation
SAGE has no reliable specimen ids. pHash-disjoint groups do not prove specimen
independence (two photos of the same leaf from different angles can be > 6 bits apart).
Same-specimen leakage between parent and test is therefore reduced, not excluded.
