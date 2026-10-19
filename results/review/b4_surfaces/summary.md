# B4: share of pixels a two-layer 2D representation must get wrong

Per rig view, the minority surface's share of leaf pixels (upper vs lower, by the adaxial normal). A two-layer
2D representation shows one layer per view, so this share is unavoidably the wrong surface. Mixed view = share > 0.1.

| leaf | mean minority share | mixed views / 18 | two-surface asset (mixed views) |
|---|---|---|---|
| Peach__Leaf_Curl__0006 | 0.106 | 6/18 | - |
| Peach__Leaf_Curl__0063 | 0.083 | 5/18 | - |
| Peach__Leaf_Curl__0087 | 0.107 | 5/18 | - |
| Tomato__Leaf_Mold__0015 | 0.066 | 5/18 | 5/18 |
| Tomato__Leaf_Mold__0052 | 0.082 | 5/18 | 6/18 |
| Tomato__Leaf_Mold__0092 | 0.132 | 8/18 | 11/18 |
| Citrus__Canker__0004 | 0.043 | 4/18 | - |
| Citrus__Canker__0012 | 0.164 | 7/18 | - |
| Citrus__Canker__0016 | 0.230 | 11/18 | - |
| Mango__Anthracnose__0044 | 0.004 | 0/18 | - |
| Mango__Anthracnose__0049 | 0.053 | 3/18 | - |
| Mango__Anthracnose__0094 | 0.040 | 3/18 | - |

Mold footprint on the lower surface = upper spot footprint by construction (KB 'below leaf spots'); only texture differs.
Leaf thickness 2*delta = 0.004 L is an assumption (bilayer.py), not a measurement.
