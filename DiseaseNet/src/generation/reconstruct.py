"""Stage 0 step 2: SAM 3D reconstruction of the parent photos (seed 0), resumable.

Same model path as SAGE-Extend (src/reconstruction/run_sam3d_batch.py: stage-wise CPU offload
so it fits a 12 GB TITAN X), but DiseaseNet inputs/outputs and the DiseaseNet leakage Guard.
Mesh export is skipped (not needed for splat rendering; saves ~8 MB/asset).

usage (sam3d env, one GPU):
  CUDA_VISIBLE_DEVICES=1 LIDRA_SKIP_INIT=true ~/micromamba/envs/sam3d/bin/python src/generation/reconstruct.py
  -> data/synthetic/stage0/assets/<asset_id>/{splat.ply, pose.npz, run_meta.json}
"""
import glob
import json
import os
import sys
import time
import traceback

os.environ.setdefault("LIDRA_SKIP_INIT", "true")
os.environ.setdefault("CONDA_PREFIX", os.path.expanduser("~/micromamba/envs/sam3d"))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.expanduser("~/Desktop/Research/external/SAGE-Extend")
SAM3D = os.path.expanduser("~/Desktop/Research/external/sam-3d-objects")
sys.path.insert(0, os.path.join(SAM3D, "notebook")); sys.path.insert(0, f"{EXT}/src/reconstruction"); sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

STAGE = os.environ.get("DN_STAGE", "stage0")


def main():
    from inference import load_image, load_single_mask
    import offload
    # leakage check before any compute: inputs must be listed by src/provenance/verify_inputs.py
    # (run in the Research .venv; this env has no parquet engine) under the CURRENT lock
    ver = json.load(open(f"{ROOT}/data/synthetic/{STAGE}/verified_inputs.json"))
    lock = json.load(open(f"{ROOT}/data/real_splits/LOCK.json"))["test_sha1_sha256"]
    assert ver["lock"] == lock, "verified_inputs.json was made under a different lock"
    inputs = sorted(glob.glob(f"{ROOT}/data/synthetic/{STAGE}/inputs/*"))
    for d in inputs:
        roots = json.load(open(f"{d}/mask_meta.json"))["root_sha1"]
        assert ver["inputs"].get(os.path.basename(d)) == roots, f"LEAKAGE GUARD: {d} not verified"
    todo = [d for d in inputs if not os.path.exists(f"{ROOT}/data/synthetic/{STAGE}/assets/{os.path.basename(d)}/run_meta.json")]
    print(f"{len(inputs)} inputs, {len(todo)} to reconstruct", flush=True)
    if not todo:
        return
    os.chdir(SAM3D)
    t0 = time.time(); pipe = offload.build_pipeline(os.path.join(SAM3D, "checkpoints/hf/pipeline.yaml")); t_load = time.time() - t0
    for d in todo:
        aid = os.path.basename(d); out = f"{ROOT}/data/synthetic/{STAGE}/assets/{aid}"; os.makedirs(out, exist_ok=True)
        offload.STAGE_TIMES.clear(); torch.cuda.reset_peak_memory_stats(); t1 = time.time()
        meta = dict(asset_id=aid, seed=0, input=d, load_s=round(t_load, 1), gpu=torch.cuda.get_device_name(0), offload=True,
                    root_sha1=json.load(open(f"{d}/mask_meta.json"))["root_sha1"])
        try:
            o = offload.run(pipe, load_image(f"{d}/image.png"), load_single_mask(d, index=0), 0)
            o["gs"].save_ply(f"{out}/splat.ply")
            pose = {k: o[k].detach().float().cpu().numpy() for k in ("rotation", "translation", "scale") if k in o}
            if "pointmap" in o:                         # needed for the input-camera intrinsics (photo-like views)
                pose["pointmap"] = o["pointmap"].float().numpy().astype(np.float16)
            np.savez_compressed(f"{out}/pose.npz", **pose)
            meta.update(status="ok", n_gaussians=int(o["gs"].get_xyz.shape[0]))
        except Exception as e:  # noqa: BLE001
            meta.update(status="failed", error=repr(e)[:2000], tb=traceback.format_exc()[-3000:]); torch.cuda.empty_cache()
        meta.update(infer_s=round(time.time() - t1, 1), peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        json.dump(meta, open(f"{out}/run_meta.json", "w"), indent=1)
        print("RUN", json.dumps({k: meta[k] for k in ("asset_id", "status", "infer_s", "peak_vram_gb")}), flush=True)


if __name__ == "__main__":
    main()
