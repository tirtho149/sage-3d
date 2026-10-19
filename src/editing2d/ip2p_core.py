"""2D editing arms (B: generic text, C: SAGE-KB text) with InstructPix2Pix.

Same backbone and guidance settings as GaussianEditor's edit-n2n guidance
(timbrooks/instruct-pix2pix, guidance_scale 7.5, image_guidance_scale 1.5),
so arms B/C vs E/F differ only in 2D vs 3D representation.

usage: ip2p.py <image.png> <mask.png> <prompt> <seed> <out.png> [--steps 20] [--res 512]
The edit is composited back inside the organ mask only (outside = original),
mirroring GaussianEditor's semantic-mask-restricted editing.
"""
import argparse, json, time, os
from scipy import ndimage
import numpy as np, torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler


DEV = os.environ.get("IP2P_DEVICE", "cuda")
_pipe = None
def pipe():
    global _pipe
    if _pipe is None:
        _pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16 if DEV == "cuda" else torch.float32, safety_checker=None).to(DEV)
        _pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(_pipe.scheduler.config)
        _pipe.set_progress_bar_config(disable=True)
    return _pipe

def edit(img: Image.Image, mask: np.ndarray, prompt: str, seed: int, steps: int = 20, res: int = 512, gs: float = 7.5, igs: float = 1.5,
         dilate: float = 0.0):
    """dilate > 0 (arm I, shape-capable 2D): the composite mask is dilated by `dilate` x the organ's equivalent
    diameter, so the editor may move the silhouette (outgrowths, deformation) instead of only recolouring inside it."""
    W, H = img.size
    s = res / max(W, H); w, h = int(W * s) // 8 * 8, int(H * s) // 8 * 8
    x = img.resize((w, h), Image.BICUBIC)
    g = torch.Generator(DEV).manual_seed(seed)
    y = pipe()(prompt, image=x, num_inference_steps=steps, guidance_scale=gs, image_guidance_scale=igs, generator=g).images[0]
    y = np.asarray(y.resize((W, H), Image.BICUBIC)).astype(np.float32)
    mb = mask > 127
    if dilate > 0:
        r = max(1, int(round(dilate * 2 * np.sqrt(mb.sum() / np.pi)))); yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        mb = ndimage.binary_dilation(mb, structure=(xx * xx + yy * yy) <= r * r)
    m = ndimage.gaussian_filter(mb.astype(np.float32), 1.0)[..., None] if dilate > 0 else mb.astype(np.float32)[..., None]
    out = y * m + np.asarray(img).astype(np.float32) * (1 - m)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))

