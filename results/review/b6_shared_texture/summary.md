# B6: shared-canonical-texture baseline (2D KB edit lifted onto one 3D asset)

n = 15 specimens. Cross-view = median same-point dE over adjacent ring views (lower = more consistent).

| method | cross-view dE (median over specimens) | edit size: colour dE vs D | silhouette change |
|---|---|---|---|
| D unedited (floor) | 2.0 | 0 | 0 |
| **Ctex: 2D KB edit lifted onto the shared asset** | **2.7** | 32.2 | 0.00 |
| per-view 2D KB edits (Cmv, E4) | 19.3 | see slide E4 | see slide E4 |
| GaussianEditor generic (E, E4) | 9.3 | see slide E4 | see slide E4 |
| GaussianEditor + KB (F, E4) | 10.4 | see slide E4 | see slide E4 |
| geometry editor (KB) (J, E4) | 2.0 | see slide E4 | see slide E4 |

Ctex needs no 3D editing and no geometry knowledge; its consistency comes from sharing one asset.
Hidden (back-facing) Gaussians keep their colour: the lifted edit covers only what the photo shows.
