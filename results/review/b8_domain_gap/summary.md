# B8: where the render-to-real gap comes from (matched pilot, 15 specimens)

Median cosine distance to the nearest real build-pool photo of the same pilot class (DINOv2 ViT-B/14), and
real-vs-condition AUC. Each row changes one factor relative to the row above.

| condition | what changes | median NN distance | AUC vs real |
|---|---|---|---|
| photo | real photo (reference) | 0.001 | 0.18 |
| in_photo | object rendered at the photo's pose, photo background | 0.276 | 0.76 |
| in_white | + white background | 0.364 | 0.96 |
| rig_white | + new viewpoints | 0.570 | 1.00 |
| rig_phototex | texture replaced by photo colours | 0.561 | 1.00 |

Step changes in median distance:
- render object at matched pose+background (photo -> in_photo): +0.275
- background (in_photo -> in_white): +0.088
- viewpoint (in_white -> rig_white): +0.206
- texture = photo colours (rig_white -> rig_phototex): -0.010
