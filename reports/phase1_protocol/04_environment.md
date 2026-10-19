# 04 — Local Environment (beast0, 2× GTX TITAN X 12 GB, Maxwell sm_52)

Every component runs locally. Nothing in this report uses Nova. Measurements are from logs in `~/Desktop/Research/logs/`.

## SAM 3D Objects (`micromamba env sam3d`)

| Item | Value |
|---|---|
| Code | facebookresearch/sam-3d-objects @ `f91db41` (`external/sam-3d-objects`) |
| Checkpoints | `facebook/sam-3d-objects` `checkpoints/hf` (13 GB), copied from Nova via Globus; gated HF access verified with the user-supplied token |
| Python / torch | 3.11 / 2.5.1+cu121 (sm_50 PTX/SASS present) |
| CUDA toolchain | conda `cuda-nvcc 12.1`, `gxx 11`, `TORCH_CUDA_ARCH_LIST=5.2` |
| Built from source for sm_52 | pytorch3d @75ebeea; gsplat @2323de5 (**patched**); cumm 0.7.11 + spconv 2.3.8 (AOT kernels, `CUMM_CUDA_ARCH_LIST=5.2`) |
| Not installable | `flash_attn` (needs sm_80+); attention runs through PyTorch `sdpa`, same as the Nova A100 log |
| Pins | numpy 1.26.4 (kaolin ABI), plyfile 1.0.3, setuptools<81 |

**Changes needed for 12 GB / Maxwell:**

1. **Stage-wise offload** (`src/reconstruction/offload.py`). The pipeline is built on CPU, and each stage moves only its own models to the GPU. Weights, dtype, sampler, and seeds are unchanged.
2. **gsplat pre-Volta patch** (`patches_gsplat_preVolta.diff`). `cooperative_groups::labeled_partition` (sm_70+) is replaced by per-thread atomics on sm<70. This only affects gradient accumulation and is numerically equivalent.
3. **spconv rebuilt from source.** The `spconv-cu121` wheel has no sm_52 kernels, and its NVRTC JIT fallback fails because headers are missing (`tensorview/core/all.h`). The source build was verified against dense `conv3d`: max abs error 2.5e-6 in fp32 and 4.9e-4 in fp16.

**Reproduction check.** The repo's leaf image, mask, and seed 42 were run locally and compared with the collaborator's A100 output:

| | A100 (Nova) | TITAN X (local) |
|---|---|---|
| Gaussians | 37,120 | 37,120 (same order) |
| Bounding box | identical to 3 d.p. | identical to 3 d.p. |
| Chamfer / bbox diagonal | — | **4.5e-5** |
| Max per-Gaussian position diff | — | 9.3e-4 (≈0.08% of diagonal) |
| Mean colour | (0.453, 0.537, 0.087) | (0.452, 0.537, 0.087) |

The differences are at the level of GPU numerical nondeterminism.

**Cost per reconstruction** (one image, one seed):

| Stage | Seconds |
|---|---|
| Sparse structure | 895 |
| SLAT | 79 |
| Decode | 23 |
| Pointmap + conditioning | 12 |
| **Total** | **1,004 s (~17 min)** |
| Weight load | +64 s, once per batch |
| Peak VRAM | **11.54 GB**, so only GPU 1 fits (GPU 0 carries the display, ~0.5 GB) |

## GaussianEditor (`micromamba env gedit`)

| Item | Value |
|---|---|
| Code | buaacyw/GaussianEditor @ `9249f84` (`external/GaussianEditor`), unmodified |
| Python / torch | 3.8 / **2.0.1+cu118** (re-pinned after a dependency silently upgraded it to 2.4.1) |
| Deps | `requirements.lock.txt` from the repo, plus constraints pinning torch/torchvision/numpy |
| Built for sm_52 | diff-gaussian-rasterization, simple-knn, nvdiffrast, tiny-cuda-nn @212104 (linked against `lib/stubs/libcuda.so`) |
| Runtime requirement | `LD_LIBRARY_PATH=$ENV/lib`, because extensions built with conda GCC 11 need `GLIBCXX_3.4.29` |
| LangSAM | lang-segment-anything @134a48a (GroundingDINO + SAM). Imported by GaussianEditor; **not used** in the pilot, because each SAM 3D asset is already the masked organ |
| Wrapper | `src/gaussian_editing/run_ge.sh` uses the collaborator's `submit_ge_edit.sh` hyperparameters verbatim; only `max_steps` and `seed` vary |
| Training views | `src/rendering/views.py`: 26 renders of the asset itself (12 azimuths × ±20° elevation, plus top and bottom) with GaussianEditor's own rasterizer, and a COLMAP export. There are no real extra views, because SAGE is single-view. |

**Why the Nova attempt failed, and the fix here:**

| Nova failure | Local fix |
|---|---|
| `tiny-cuda-nn` "metadata-generation-failed" | Build with conda nvcc 11.8 and `-lcuda` from the stubs dir |
| `make_colmap_from_ply.py`: `No module named sam3d_objects.pipeline.gaussian` | Views are now rendered in the GaussianEditor env with its own `GaussianModel`/renderer, so the SAM 3D package isn't needed there |

**Cost:** measured in the smoke test (see `06_pilot_findings.md`).

## 2D editor (arms B/C)

`src/editing2d/ip2p.py` runs `timbrooks/instruct-pix2pix` with Euler-ancestral sampling for 20 steps, guidance 7.5, and image guidance 1.5. These match GaussianEditor's IP2P guidance. It runs in the main `.venv` (torch 2.6.0+cu124, diffusers 0.41), at 3.1 GB peak VRAM.

## Disk

The root filesystem stands at 13–15 GB free. The micromamba environments total about 25 GB and the SAGE shards 20 GB. Each pilot reconstruction is about 5 MB. Regenerable caches were cleared during setup (pip, micromamba).
