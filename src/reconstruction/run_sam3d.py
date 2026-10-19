"""Run SAM 3D Objects on one input dir (image.png + 0.png). Records runtime and peak VRAM.

usage: run_sam3d.py <input_dir> <out_dir> <seed>
Outputs: splat.ply, mesh.obj (if produced), run_meta.json
"""
import os, sys, time, json
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
SAM3D = os.path.expanduser("~/Desktop/Research/external/sam-3d-objects")
sys.path.insert(0, os.path.join(SAM3D, "notebook"))
os.chdir(SAM3D)
import torch
from inference import load_image, load_single_mask
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offload

inp, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(out, exist_ok=True)
t0 = time.time()
pipe = offload.build_pipeline(os.path.join(SAM3D, "checkpoints/hf/pipeline.yaml"))
t_load = time.time() - t0
image = load_image(os.path.join(inp, "image.png"))
mask = load_single_mask(inp, index=0)
torch.cuda.reset_peak_memory_stats()
t1 = time.time()
output = offload.run(pipe, image, mask, seed)
t_inf = time.time() - t1
output["gs"].save_ply(os.path.join(out, "splat.ply"))
# pose (object-local -> camera, PyTorch3D camera convention) + MoGe camera-space pointmap, for input-view rendering (E1)
import numpy as np
pose = {k: output[k].detach().float().cpu().numpy() for k in ("rotation", "translation", "scale") if k in output}
if "pointmap" in output:
    pose["pointmap"] = output["pointmap"].float().numpy().astype(np.float16)
np.savez_compressed(os.path.join(out, "pose.npz"), **pose)
meta = dict(input=inp, seed=seed, load_s=round(t_load, 1), infer_s=round(t_inf, 1),
            peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), gpu=torch.cuda.get_device_name(0), stage_s={k: round(v, 1) for k, v in offload.STAGE_TIMES.items()}, offload=True,
            n_gaussians=int(output["gs"].get_xyz.shape[0]) if hasattr(output["gs"], "get_xyz") else None)
if "mesh" in output:
    import trimesh
    ms = output["mesh"] if isinstance(output["mesh"], list) else [output["mesh"]]
    m = ms[0]
    trimesh.Trimesh(vertices=m.vertices.cpu().numpy(), faces=m.faces.cpu().numpy()).export(os.path.join(out, "mesh.obj"))
json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)
print("RUN_META", json.dumps(meta))
