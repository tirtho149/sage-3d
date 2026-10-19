# Demo summary (specimen = unit; seeds averaged within specimen)

## Tomato Leaf Mold: where the mold-coloured pixels land

| method | mold on UPPER | mold on LOWER | upper spots on LOWER | footprint on spots | n |
|---|---|---|---|---|---|
| REAL_photo | 0.002 | - | - | - | 3 |
| B_2D_generic | 0.004 | - | - | - | 3 |
| C_2D_kb_joint | 0.410 | - | - | - | 3 |
| Cstage_2D_kb | 0.367 | - | - | - | 3 |
| D | 0.029 | 0.034 | 0.156 | - | 3 |
| E | 0.017 | 0.026 | 0.118 | - | 3 |
| F | 0.053 | 0.052 | 0.171 | - | 3 |
| K | 0.018 | 0.019 | 0.158 | - | 3 |
| BL0 | 0.033 | 0.038 | 0.045 | - | 3 |
| BLk | 0.038 | 0.168 | 0.047 | 0.798 | 3 |
| BLr | 0.039 | 0.131 | 0.047 | 0.145 | 3 |
| BLu | 0.164 | 0.043 | 0.043 | - | 3 |

## Mango leaves + 'Leaves curl upwards' (direction control)

| method | roll (deg) | kappa_t*w | strain p95 | relief corr | input IoU | n |
|---|---|---|---|---|---|---|
| D | 4.8 | 0.17 | 0.000 | 1.000 | 0.912 | 3 |
| G | 4.8 | 0.17 | 0.066 | 0.712 | 0.915 | 3 |
| CU_kb | 46.2 | 2.17 | 0.110 | 0.487 | 0.623 | 3 |
| CD_opposite | -35.4 | -1.54 | 0.107 | 0.435 | 0.684 | 3 |

direction: KB-direction edits raised the margins in 3/3 specimens; the opposite-direction control lowered them in 3/3.

## Peach Leaf Curl

| method | roll (deg) | strain p95 | relief corr | input IoU | pucker enrichment on reddish tissue | n |
|---|---|---|---|---|---|---|
| D | 0.6 | 0.000 | 1.000 | 0.707 | 0.00 | 3 |
| J | -1.3 | 0.626 | 0.707 | 0.702 | 1.15 | 3 |
| G | 0.8 | 0.182 | 0.945 | 0.689 | 1.18 | 3 |
| E | - | - | - | 0.700 | - | 3 |
| F | - | - | - | 0.702 | - | 3 |
| K | 0.6 | 0.000 | 1.000 | 0.707 | 0.00 | 3 |
| H | -1.3 | 0.626 | 0.707 | 0.702 | 1.15 | 3 |
| J2 | -1.0 | 0.601 | 0.735 | 0.703 | 1.14 | 3 |
| J2u | -0.0 | 0.611 | 0.717 | 0.700 | 1.07 | 3 |
| J2+curl_up_HYP | 20.3 | 0.745 | 0.426 | 0.621 | 1.24 | 3 |
| J2+curl_down_HYP | -21.2 | 0.759 | 0.499 | 0.615 | 1.08 | 3 |

## Corn Smut

| method | galls on kernel tissue | total gall volume (cm^3, assumed scale) | undercut fraction | input IoU | n |
|---|---|---|---|---|---|
| D | - | - | - | 0.870 | 3 |
| J | - | - | 0.00 | 0.790 | 3 |
| G | - | - | 0.00 | 0.831 | 3 |
| E | - | - | - | 0.862 | 3 |
| F | - | - | - | 0.861 | 3 |
| V | 1.00 | 35.7 | 0.38 | 0.840 | 2 |
| Vr | 0.22 | 35.7 | 0.38 | 0.841 | 2 |

height-field J added volume (cm^3, same scale): {"Corn__Smut__0012": 33.88, "Corn__Smut__0052": 18.97}
status: {"Corn__Smut__0012": "ok", "Corn__Smut__0052": "ok", "Corn__Smut__0025": "ABSTAIN: no kernel tissue identifiable in this reconstruction (the evidence-based placement rule cannot be applied)"}
