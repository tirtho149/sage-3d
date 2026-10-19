# 11: Search for public multi-view / 3D data on diseased plant organs

Date of search: 2026-10-14. Search tools: web search and direct page fetches.
Scope: public data that could validate three claims the SAGE-Extend 3D pipeline makes beyond a single photograph:

- **(a) Underside.** A symptom predicted on the abaxial (lower) surface, for example Tomato Leaf Mold sporulation, matches a real photograph of that same leaf's underside.
- **(b) Curl / 3D shape.** The predicted curl direction and curvature, for example Peach Leaf Curl, match a 3D measurement of the same leaf.
- **(c) Gall volume.** The predicted gall count, size and volume, for example Corn Common Smut, match physical measurements of the same specimen.

To count as validation data, a dataset must hold several views or a 3D measurement **of the same specimen** that was also photographed in a SAGE-like view. Large datasets of unrelated single photographs do not meet this condition.

Verification legend:
- **V**: the stated fact was read on the dataset page, the paper, or the PDF text during this search.
- **S**: the fact comes only from a search-engine summary or a secondary page; the primary source was not opened or could not be read.
- **U**: not verified. Release status or licence could not be confirmed.

## 1. Queries run

| # | Query (abridged) | Useful hits |
|---|---|---|
| 1 | multi-view 3D dataset diseased plant leaves point cloud disease | UW-MRDC 3D wheat, PLANesT-3D, GaussianPlant, PAA-Plant |
| 2 | Pheno4D maize tomato point clouds license | Pheno4D |
| 3 | dataset leaf images both sides adaxial abaxial same leaf disease | coffee leaf (abaxial only), tree-leaf sets |
| 4 | NeRF / 3D Gaussian splatting plant disease dataset multi-view | GaussianPlant, Nottingham wheat 3DGS, NIRPlant |
| 5 | leaf curl 3D measurement point cloud leaf rolling drought maize | MaizeField3D, lidar drought studies (no curl data) |
| 6 | corn smut gall 3D imaging Ustilago maydis ear volume | none (agronomic gall-size studies only) |
| 7 | hyperspectral 3D point cloud diseased plants sugar beet Cercospora | Roscher et al. 2016 (method), BonnBeetClouds3D |
| 8 | UW-MRDC 3D wheat Fusarium head blight multispectral point cloud | UW-MRDC record, arXiv 2303.05634 |
| 9 | tomato yellow leaf curl virus 3D point cloud leaf curling | TYLCV RGB-D greenhouse study (data status U) |
| 10 | Soybean-MVS multi-view stereo soybean | Soybean-MVS |
| 11 | Crops3D point clouds license | Crops3D |
| 12 | maize ear 3D point cloud multi-view public | SC-NeRF, maize ear-rot MVS paper |
| 13 | "leaf curl" peach Taphrina deformans image dataset | none (extension photos only) |
| 14 | grapevine downy mildew abaxial sporulation image dataset | INSA Lyon hyperspectral vine mildew, Merlot DM set |
| 15 | "both sides" leaf "upper and lower" surfaces same leaf | SoyCultivar (arXiv 1910.04919), Costa Rica leaf sides |
| 16 | X-ray CT plant galls root-knot nematode clubroot gall volume | none (root CT methods only) |
| 17 | ROSE-X rosebush X-ray tomography | ROSE-X |
| 18 | Plant3D Arabidopsis tomato tobacco laser scan | Plant3D (Dryad / Mendeley) |
| 19 | soybean leaf front back reflective transparent 200 cultivars | SoyCultivar details |
| 20 | 3D reconstruction diseased leaf lesion mapping apple scab powdery mildew | AppleScab-LT (2D only) |
| 21 | wheat leaf rust both leaf surfaces adaxial abaxial uredinia | none with surface labels |
| 22 | tomato leaf mold Passalora fulva underside image dataset | Roboflow project (undescribed) |
| 23 | maize ear rot multi-view stereo point cloud (Comput. Electron. Agric. 2023) | title and venue only |
| 24 | LAST-Straw strawberry 3D point clouds | LAST-Straw |
| 25 | rice leaf rolling 3D curvature point cloud | "Rolled-upness" method paper (2D cross-sections) |
| 26 | "multi-view" plant disease image dataset same leaf multiple angles | PlantVillage (in-plane rotations), tomato/cotton sets |
| 27 | corn smut image dataset Kaggle Mendeley | scattered photo libraries only |
| 28 | single leaf 3D scan dataset curvature ground truth structured light | none plant-specific |
| 29 | adaxial abaxial paired cucumber downy mildew soybean rust | ASDID, cucumber (surface not stated) |

Pages opened directly: arXiv 2303.05634, ipb.uni-bonn.de/data/pheno4d, IEEE DataPort GaussianPlant, HF PAA-Plant-Dataset, arXiv 2503.21958, INSA Lyon vine-mildew sheet (PDF text), arXiv 1907.11561 (PDF text), arXiv 2407.21150 (PLANesT-3D), HF BGLab/AgriField3D, github.com/clawCa/Crops3D, arXiv 1511.08060 (PDF text), Frontiers 10.3389/fpls.2016.01419, AAAI NIRPlant PDF, arXiv 2604.03328.
Pages that failed: Nottingham repository (HTTP 403), PMC Pheno4D article (bot check), AMiner ear-rot record (no content).

## 2. Candidate table

Column key: "Same specimen" means the views or measurements belong to one physical specimen. "Both surf." means the adaxial and abaxial surfaces of the same leaf were photographed. Verdict letters: (a) underside, (b) curl / 3D shape, (c) gall volume, (d) none. "Prior only" means the dataset can supply healthy-shape priors or test the evaluation code, but it cannot validate a disease claim.

### 2.1 Diseased plant organs

| Dataset | URL | Licence | Modality | Species / disease | # specimens | Same specimen | Both surf. | Poses | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| PlantVillage (Hughes & Salathé 2015) | arxiv.org/abs/1511.08060 | CC BY-NC-SA 4.0 on arXiv page (V); mirrors vary | RGB; "4–7 images" per detached leaf, leaf "rotated around 360 degrees", leaf against grey/black paper (V) | 14 crops, 54,309 images (V); includes a Tomato Leaf Mold class (class list from prior knowledge of the release, not re-opened: S) | 54,309 images; grouping by leaf for 41,112 (V, Frontiers 2016) | Yes, but views are in-plane rotations of a flat leaf | Not stated (U) | No | **(d)**. In-plane rotations add no new geometry or surface. |
| UW-MRDC 3D Wheat (Hamila et al.) | arxiv.org/abs/2303.05634; DOI 10.57702/s3yfrrz1 (S) | Paper CC BY 4.0; dataset licence U | Multispectral 3D scanner, RGB+NIR point clouds (V) | Wheat, Fusarium head blight vs healthy (V) | ">300" (V); 216 / 576 elsewhere (S) | Yes (3D scan) | n/a (heads) | U | **(d)** for our diseases. The only public diseased-plant 3D set found; it covers head colour, not curl, underside or galls. |
| Vine downy mildew hyperspectral (INSA Lyon) | bioinfo.insa-lyon.fr/sdata/mildew_hyperspectral_images/ | Not stated on sheet (U) | Hyperspectral line scanner (Specim FX10, 400–1000 nm), single view (V) | *Vitis vinifera* Cabernet Sauvignon, *Plasmopara viticola*, inoculated on abaxial side (V) | 165 images; replicate = same leaf over time (V) | Yes (time series) | Imaged surface not stated (U) | No | **(d)**. It would be (a)-adjacent only if both surfaces were imaged, and the sheet does not say so. |
| Coffee leaf biotic stress (Esgario et al. 2019) | arxiv.org/abs/1907.11561 | U | Smartphone RGB, white background (V) | Coffee: leaf miner, rust, brown leaf spot, Cercospora (V) | 1,747 images (V) | No | **Abaxial only** (V) | No | **(d)**. Underside photographs exist, but there are no paired adaxial views. |
| AppleScab-LT | arxiv.org/abs/2608.14235 (S) | U | 2D RGB, same leaves over a season, pixel masks (S) | Apple scab | U | Yes (time) | U | No | **(d)** |
| Roscher et al. 2016, hyperspectral 3D plant models | isprs-annals.copernicus.org/articles/III-7/89/2016/ (S) | U | Hyperspectral + 3D geometry (S) | Sugar beet, Cercospora leaf spot | U | Yes | U | U | **(d)**. Method paper; data release U. |
| TYLCV resistance from RGB-D (Shmaryahu et al.) | cris.bgu.ac.il (S) | U | RGB-D, greenhouse canopy (S) | Tomato, *Tomato yellow leaf curl virus* | U | U | No | U | **(d)** as found. Leaf-level curvature not reported (S); data status U. A curl-relevant lead worth a request to the authors. |
| Maize ear rot, MVS + point cloud segmentation (Comput. Electron. Agric. 2023) | aminer record (title only) (S) | U | Multi-view stereo point clouds (S) | Maize ear rot | U | Yes (S) | n/a | U | **(d)** as found. Closest analogue to (c) (a diseased ear in 3D), but not galls; data status U. |
| Merlot downy mildew (Abdelghafour et al. 2021) | data.mendeley.com/datasets/yxf2yv5ymt (S) | CC BY-NC-ND 4.0 (S) | Whole-vine field RGB (S) | Grape downy mildew | 99 images (S) | No | No | No | **(d)** |
| Tomato Leaf Mold (Roboflow) | universe.roboflow.com/anfal/tomato-leaf-mold (S) | CC BY 4.0 (S) | RGB detection, undescribed | Tomato Leaf Mold | U | U | U | No | **(d)** |
| ASDID soybean (Dryad) | datadryad.org/dataset/doi:10.5061/dryad.41ns1rnj3 (S) | U | RGB, attached or detached leaves (S) | Soybean incl. rust, downy mildew | 9,981 images (S) | No | Not stated | No | **(d)** |
| Corn smut photo libraries (Biolib, BioImages, EPPO, Wikimedia) | e.g., bioimages.org.uk/image.php?id=145411 (S) | Mixed: CC BY-SA, CC BY-NC-SA, educational only (S) | Single photographs | *Ustilago maydis* galls | Scattered | No | n/a | No | **(d)** |

### 2.2 Both-surface leaf photography (healthy)

| Dataset | URL | Licence | Modality | Species | # specimens | Same specimen | Both surf. | Poses | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| SoyCultivar (arXiv 1910.04919) | arxiv.org/abs/1910.04919 | Release and licence U | Flatbed scanner, 600 dpi, reflective + transmissive, front + back (S, from search summary of paper) | Soybean, 200 cultivars, healthy | 1,200 leaves (S) | Yes | **Yes** (S) | n/a (flat scan) | **(d)** for disease. If released, it would be a both-surface registration testbed for healthy leaves. |
| Costa Rica leaf-sides (Carranza-Rojas & Mata-Montero 2016) | repositoriotec.tec.ac.cr/handle/2238/9223 (S) | U | Photographs of front and underside (S) | 63 tree species, healthy | U | U | Yes (S) | No | **(d)** |

### 2.3 Multi-view or 3D plants (healthy; shape priors and evaluation-code testbeds)

| Dataset | URL | Licence | Modality | Species | # specimens | Same specimen | Poses | Verdict |
|---|---|---|---|---|---|---|---|---|
| PAA-Plant-Dataset | huggingface.co/datasets/JackTrum938/PAA-Plant-Dataset | Apache-2.0 (V) | 102 RGB views per sample + COLMAP sparse model + scanner point cloud ("HyperScan") + masks (V) | Broccoli, cabbage, cotton, rice, sunflower, tobacco (V) | 41 samples (V) | Yes | COLMAP (V) | **Prior only.** The best public testbed found for held-out-view and point-cloud metrics on real plants, because it pairs real multi-view images with an independent scan. |
| PLANesT-3D | github.com/visionlab-ogu/PLANesT-3D; aperta.ulakbim.gov.tr/record/286354 | Dataset licence U (paper CC BY-NC-ND 4.0) | Handheld DSLR, 150–406 images per plant, Metashape SfM/MVS; images and camera poses released (V) | Pepper 10, rose 10, ribes 14 (V) | 34 (V) | Yes | Yes (V) | **Prior only.** Real leaves with released poses. |
| Pheno4D | ipb.uni-bonn.de/data/pheno4d/ | Not stated on page (V); a repository record lists "GRO License" (S) | Laser-scanned point clouds (scanner model not read: U) | 7 maize (12 days), 7 tomato (20 days) (V) | 224 clouds; 126 labelled (V) | Yes (time series) | n/a | **Prior only** (tomato and maize leaf geometry). |
| Plant3D (Conn et al. 2017) | datadryad.org/dataset/doi:10.6075/J0DZ06NT | U | Laser scans, no colour (S) | Arabidopsis, tomato, tobacco, sorghum; light/heat/shade/drought (S) | 559–714 (sources differ, S) | Yes | n/a | **Prior only** |
| Crops3D | github.com/clawCa/Crops3D; doi.org/10.6084/m9.figshare.27313272 | Not on GitHub page (V); CC BY 4.0 per Voxel51 (S) | Point clouds from several methods (LiDAR, structured light, MVS: S) | 8 crops incl. maize, tomato (V) | 1,230 (V) | Yes | U | **Prior only** |
| MaizeField3D / AgriField3D | huggingface.co/datasets/BGLab/AgriField3D | CC BY-NC 4.0 (V) | Terrestrial laser scanner; NURBS leaf surfaces (V) | Field maize, diversity panel (V) | 1,045 raw, 520 segmented (V) | Yes | n/a | **Prior only** (maize leaves; cob presence flagged, but no ear geometry or galls) |
| Soybean-MVS | mdpi.com/2077-0472/13/7/1321 (S) | U | RGB MVS (S) | Soybean, 5 varieties, 13 stages (S) | 102 (S) | Yes | U | **Prior only** |
| ROSE-X | plantmethods.biomedcentral.com/articles/10.1186/s13007-020-00573-w (S) | U | X-ray CT, voxel labels (S) | Rose | 11 (S) | Yes | n/a | **Prior only** |
| LAST-Straw | arxiv.org/abs/2403.00566 (S) | Paper CC BY 4.0; dataset U | Structured light (Einscan Pro 2X) (S) | Strawberry | 84 clouds (S) | Yes | n/a | **Prior only** |
| GaussianPlant | ieee-dataport.org/documents/gaussianplant-dataset | DataPort subscription; licence not stated (V) | >120 views per scene, COLMAP poses (V) | **Artificial** plants (V) | 10 (V) | Yes | Yes | **(d)**. Artificial plants. |
| NIRPlant (AAAI) | ojs.aaai.org/index.php/AAAI/article/view/37259 | U | RGB + NIR + text; turntable; 360 viewpoints per scene (V, PDF) | "up to 10" species incl. artificial controls (V) | 34 scenes (V) | Yes | SfM (V) | **Prior only**. Disease not mentioned. |
| Nottingham wheat 3DGS/NeRF data | nottingham-repository.worktribe.com/output/45435468 (403) | U | Robot arm + turntable (S) | Wheat | 20 plants × 6 time points (S) | Yes | U | **Prior only** (U) |
| SC-NeRF (Ku et al., arXiv 2503.21958) | arxiv.org/abs/2503.21958 | Paper CC BY-SA 4.0; data U | Stationary camera + rotating object, COLMAP (V) | "plant objects"; a maize ear per search summary (S) | U | Yes | Yes | **Prior only** (U). An ISU-affiliated group; ask the authors directly about healthy-ear captures. |

### 2.4 Method references (no data, but they define metrics)

- "Rolled-upness" (Plant Methods 2015, PMC4650205; S) gives an objective leaf-rolling curvature measure from cross-sections. It informs the curl metric in report 12.
- Review of point-cloud leaf-surface reconstruction (arXiv 2604.03328; V abstract) evaluates on LAST-Straw, Pheno4D and Crops3D. It is useful for choosing a surface-fitting method on the reference clouds.

## 3. Conclusion

**(a) Underside validation: public data is not sufficient.** No public dataset was found in which the adaxial and abaxial surfaces of the same *diseased* leaf were both photographed and released. The two both-surface sets found (SoyCultivar, Costa Rica) are healthy and their release is unverified. The underside-symptom sets (coffee, vine mildew) photograph one surface only, or do not state which.

**(b) Curl / 3D shape validation: public data is not sufficient.** No public 3D measurements of disease-induced leaf curl or roll were found for Peach Leaf Curl or any other disease. The TYLCV RGB-D study is canopy-level and its data status is unverified. Healthy 3D leaf datasets (PAA-Plant, PLANesT-3D, Pheno4D, Crops3D, MaizeField3D, Plant3D) can supply curvature priors for healthy controls. PAA-Plant and PLANesT-3D, which both release real images and poses, can be used now to test the held-out-view evaluation code. Neither kind of data can show that a KB-driven curl edit matches a real diseased leaf.

**(c) Gall-volume validation: public data is not sufficient.** No 3D or volumetric data on corn smut galls, or on any plant gall, was found. The 2023 maize ear-rot MVS paper is the nearest analogue, a diseased ear in 3D, but its data status is unverified and ear rot does not produce galls.

**Implication.** The 3D claims (underside, curl, gall volume) cannot be tested with public data. A small purpose-built capture is required; report 12 specifies it. Two low-cost steps can start before capture:
1. Run the held-out-view metric code on PAA-Plant-Dataset (Apache-2.0) and PLANesT-3D to verify it and to measure the noise floor of the MVS reference.
2. Write to the authors of the TYLCV RGB-D, maize ear-rot MVS, SoyCultivar and SC-NeRF papers asking about data access. Until a reply arrives, record each of these as unverified.

The search was limited to English-language web sources reachable without login. Datasets behind institutional portals, or described only in non-English literature, may have been missed.
