"""Stage-wise CPU<->GPU offloading for SAM 3D Objects on a 12 GB GPU.

Meta's pipeline instantiates all sub-models on the GPU (~13 GB of fp32 weights,
official requirement 32 GB VRAM). Here the pipeline is built on CPU, the compute
device is switched to CUDA, and each pipeline stage moves only its own models to
the GPU for the duration of the call. Inference maths is unchanged (same weights,
same dtype/autocast, same sampler); only residency changes, so outputs should
match a resident run up to GPU nondeterminism. Verified against the original
A100 splat in results/sam3d_repro (see reports/04_environment.md).
"""
import os, functools, time
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

STAGE_TIMES: dict[str, float] = {}




def _mods(obj):
    if isinstance(obj, torch.nn.Module):
        return [obj]
    m = getattr(obj, "model", None)
    return [m] if isinstance(m, torch.nn.Module) else []


def _wrap(pipe, name, get_modules):
    fn = getattr(pipe, name)

    @functools.wraps(fn)
    def wrapped(*a, **k):
        mods = [m for g in get_modules() for m in _mods(g)]
        for m in mods:
            m.to("cuda")
        t = time.time()
        try:
            return fn(*a, **k)
        finally:
            torch.cuda.synchronize()
            STAGE_TIMES[name] = STAGE_TIMES.get(name, 0.0) + time.time() - t
            for m in mods:
                m.to("cpu")
            torch.cuda.empty_cache()

    setattr(pipe, name, wrapped)


def build_pipeline(config_file: str):
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(config_file))), "notebook"))
    from inference import check_hydra_safety, WHITELIST_FILTERS, BLACKLIST_FILTERS
    config = OmegaConf.load(config_file)
    config.rendering_engine = "pytorch3d"
    config.compile_model = False
    config.workspace_dir = os.path.dirname(config_file)
    config.device = "cpu"
    check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
    pipe = instantiate(config)
    pipe.device = torch.device("cuda")
    M = pipe.models
    emb = lambda: [e for e in pipe.condition_embedders.values() if e is not None]
    _wrap(pipe, "compute_pointmap", lambda: [pipe.depth_model])
    _wrap(pipe, "get_condition_input", emb)
    _wrap(pipe, "sample_sparse_structure", lambda: [M["ss_generator"], M["ss_decoder"], M["ss_encoder"]] if M["ss_encoder"] is not None else [M["ss_generator"], M["ss_decoder"]])
    _wrap(pipe, "sample_slat", lambda: [M["slat_generator"]])
    _wrap(pipe, "decode_slat", lambda: [M["slat_decoder_gs"], M["slat_decoder_gs_4"], M["slat_decoder_mesh"]])
    return pipe


def run(pipe, image, mask, seed):
    import numpy as np
    rgba = np.concatenate([image[..., :3], (mask.astype(np.uint8) * 255)[..., None]], axis=-1)
    return pipe.run(rgba, None, seed, stage1_only=False, with_mesh_postprocess=False, with_texture_baking=False,
                    with_layout_postprocess=False, use_vertex_color=True, stage1_inference_steps=None, pointmap=None)
