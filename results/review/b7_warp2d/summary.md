# B7: best-case 2D warp (projection of the 3D curl) vs 3D, for the cited curl direction

dE up-vs-down = mean colour difference between the 'curl up' and 'curl down' results (object pixels).
A representation that cannot tell the two directions apart scores ~0.

| leaf | 2D warp, input view | 3D, input view | 3D, mean over 18 rig views | 2D warp IoU vs real mask (up / down) |
|---|---|---|---|---|
| Mango__Anthracnose__0044 | 13.6 | 47.2 | 45.0 | 0.59 / 0.61 |
| Mango__Anthracnose__0049 | 4.7 | 11.7 | 46.2 | 0.67 / 0.76 |
| Mango__Anthracnose__0094 | 7.8 | 31.6 | 41.3 | 0.66 / 0.73 |
