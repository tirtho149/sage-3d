"""2D editing arms (B: generic text, C: SAGE-KB text) with InstructPix2Pix.

Same backbone and guidance settings as GaussianEditor's edit-n2n guidance
(timbrooks/instruct-pix2pix, guidance_scale 7.5, image_guidance_scale 1.5),
so arms B/C vs E/F differ only in 2D vs 3D representation.

usage: ip2p.py <image.png> <mask.png> <prompt> <seed> <out.png> [--steps 20] [--res 512]
The edit is composited back inside the organ mask only (outside = original),
mirroring GaussianEditor's semantic-mask-restricted editing.
"""
import argparse, json, time
import numpy as np, torch
from PIL import Image
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

ap = argparse.ArgumentParser()
ap.add_argument("image"); ap.add_argument("mask"); ap.add_argument("prompt"); ap.add_argument("seed", type=int); ap.add_argument("out")
ap.add_argument("--steps", type=int, default=20); ap.add_argument("--res", type=int, default=512)
ap.add_argument("--gs", type=float, default=7.5); ap.add_argument("--igs", type=float, default=1.5)
a = ap.parse_args()

_pipe = None
def pipe():
    global _pipe
    if _pipe is None:
        _pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.float16, safety_checker=None).to("cuda")
        _pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(_pipe.scheduler.config)
        _pipe.set_progress_bar_config(disable=True)
    return _pipe

def edit(img: Image.Image, mask: np.ndarray, prompt: str, seed: int, steps: int, res: int, gs: float, igs: float):
    W, H = img.size
    s = res / max(W, H); w, h = int(W * s) // 8 * 8, int(H * s) // 8 * 8
    x = img.resize((w, h), Image.BICUBIC)
    g = torch.Generator("cuda").manual_seed(seed)
    y = pipe()(prompt, image=x, num_inference_steps=steps, guidance_scale=gs, image_guidance_scale=igs, generator=g).images[0]
    y = np.asarray(y.resize((W, H), Image.BICUBIC)).astype(np.float32)
    m = (mask > 127).astype(np.float32)[..., None]
    out = y * m + np.asarray(img).astype(np.float32) * (1 - m)
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))

if __name__ == "__main__":
    t = time.time()
    img = Image.open(a.image).convert("RGB"); mask = np.asarray(Image.open(a.mask).convert("L"))
    edit(img, mask, a.prompt, a.seed, a.steps, a.res, a.gs, a.igs).save(a.out)
    json.dump(dict(prompt=a.prompt, seed=a.seed, steps=a.steps, res=a.res, gs=a.gs, igs=a.igs, sec=round(time.time() - t, 1),
                   peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2)), open(a.out + ".json", "w"), indent=1)
