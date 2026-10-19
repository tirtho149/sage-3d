"""Batch SAM 3D (arm D source) over pilot specimens x seeds. Loads weights once; resumable.

usage: run_sam3d_batch.py <pilot.yaml> <results_root> [--only SUBSTR]
Writes <results_root>/<specimen>/seed<k>/{splat.ply, mesh.obj, pose.npz, run_meta.json}
Refuses any specimen not in the build split (leakage guard).
"""
import os, sys, time, json, argparse, traceback
os.environ.setdefault("LIDRA_SKIP_INIT", "true")
SAM3D = os.path.expanduser("~/Desktop/Research/external/sam-3d-objects")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(SAM3D, "notebook")); sys.path.insert(0, HERE)
import numpy as np, torch, yaml, pandas as pd
from inference import load_image, load_single_mask
import offload

ap = argparse.ArgumentParser(); ap.add_argument("cfg"); ap.add_argument("out"); ap.add_argument("--only", default="")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.cfg))
man = pd.read_csv(os.path.join(ROOT, "data/metadata/specimen_manifest.csv")).set_index("specimen_id")
jobs = [(s, k) for k in cfg["seeds"] for s in cfg["specimens"] if a.only in s]   # seed-major: a full single-seed set first
for s, _ in jobs:
    assert man.loc[s, "split"] == "build", f"LEAKAGE GUARD: {s} is in split {man.loc[s, 'split']}"
todo = [(s, k) for s, k in jobs if not os.path.exists(os.path.join(a.out, s.replace("/", "__"), f"seed{k}", "run_meta.json"))]
print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
if not todo:
    sys.exit(0)
os.chdir(SAM3D)
t0 = time.time(); pipe = offload.build_pipeline(os.path.join(SAM3D, "checkpoints/hf/pipeline.yaml")); t_load = time.time() - t0
for s, k in todo:
    inp = os.path.join(ROOT, "data/inputs", s.replace("/", "__")); out = os.path.join(a.out, s.replace("/", "__"), f"seed{k}")
    os.makedirs(out, exist_ok=True)
    offload.STAGE_TIMES.clear(); torch.cuda.reset_peak_memory_stats(); t1 = time.time()
    meta = dict(specimen=s, seed=k, input=inp, load_s=round(t_load, 1), gpu=torch.cuda.get_device_name(0), offload=True)
    try:
        image = load_image(os.path.join(inp, "image.png")); mask = load_single_mask(inp, index=0)
        o = offload.run(pipe, image, mask, k)
        o["gs"].save_ply(os.path.join(out, "splat.ply"))
        pose = {kk: o[kk].detach().float().cpu().numpy() for kk in ("rotation", "translation", "scale") if kk in o}
        if "pointmap" in o:
            pose["pointmap"] = o["pointmap"].float().numpy().astype(np.float16)
        np.savez_compressed(os.path.join(out, "pose.npz"), **pose)
        if "mesh" in o:
            import trimesh
            m = (o["mesh"] if isinstance(o["mesh"], list) else [o["mesh"]])[0]
            trimesh.Trimesh(vertices=m.vertices.cpu().numpy(), faces=m.faces.cpu().numpy()).export(os.path.join(out, "mesh.obj"))
        meta.update(status="ok", n_gaussians=int(o["gs"].get_xyz.shape[0]))
    except Exception as e:
        meta.update(status="failed", error=repr(e)[:2000], tb=traceback.format_exc()[-4000:])
        torch.cuda.empty_cache()
    meta.update(infer_s=round(time.time() - t1, 1), peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                stage_s={kk: round(v, 1) for kk, v in offload.STAGE_TIMES.items()})
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=1)
    print("RUN", json.dumps({kk: meta[kk] for kk in ("specimen", "seed", "status", "infer_s", "peak_vram_gb")}), flush=True)
