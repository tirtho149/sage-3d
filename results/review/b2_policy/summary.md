# B2: observed-evidence policy applied to all demo edits

TAU = 0.046 (max fitted-IoU drop over 24 appearance-only edits (E/F/K)). Preserve mode refuses an edit whose fitted input-view IoU
drops by more than TAU. Counterfactual mode would keep it, labelled. Hypotheses the KB does not
specify are abstentions by protocol.

| demo | edit | n | accept @TAU | refuse @TAU | abstain | accept @2TAU | median IoU drop |
|---|---|---|---|---|---|---|---|
| corn | KB height field | 3 | 1 | 2 | 0 | 1 | 0.114 |
| corn | generic geometry | 3 | 2 | 1 | 0 | 3 | 0.033 |
| corn | volumetric galls on kernels | 6 | 2 | 4 | 0 | 3 | 0.078 |
| corn | volumetric galls, random sites | 6 | 1 | 5 | 0 | 3 | 0.081 |
| mango_curl | KB curl (cited dir.) | 9 | 0 | 9 | 0 | 0 | 0.279 |
| mango_curl | generic geometry | 3 | 3 | 0 | 0 | 3 | -0.003 |
| mango_curl | reversed curl (control) | 9 | 0 | 9 | 0 | 0 | 0.205 |
| peach | KB geometry v1 | 3 | 2 | 1 | 0 | 3 | 0.012 |
| peach | KB pucker, evidence-placed | 9 | 9 | 0 | 0 | 9 | -0.008 |
| peach | KB pucker, uniform | 9 | 9 | 0 | 0 | 9 | -0.005 |
| peach | curl down (hypothesis) | 3 | 0 | 0 | 3 | 0 | 0.098 |
| peach | curl up (hypothesis) | 3 | 0 | 0 | 3 | 0 | 0.079 |
| peach | generic geometry | 3 | 2 | 1 | 0 | 3 | -0.001 |
| tomato | two-surface mold | 9 | 9 | 0 | 0 | 9 | -0.000 |
| tomato | two-surface, random | 9 | 9 | 0 | 0 | 9 | -0.000 |
| tomato | two-surface, uniform | 9 | 9 | 0 | 0 | 9 | -0.000 |
