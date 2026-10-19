"""Post-hoc object-masking of the pilot's per-view 2D edits (Bmv/Cmv), which were run with mask=None (audit F-2D-2).
ip2p_core.edit runs IP2P on the full image and composites inside the mask, so compositing an unmasked output with the
mask is EXACTLY the masked edit (same seed -> same IP2P output). Mask = alpha > 0.5 of the D render of that view.
Writes results/pilot/<sid>/{Bmv,Cmv}_masked/NNN.png. usage: mask_perview.py <sid> [<sid> ...]"""
import os, sys
import numpy as np
from PIL import Image
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, f"{R}/src")
from rendering.rig_render import load_view
for sid in sys.argv[1:]:
    for arm in ("Bmv", "Cmv"):
        src, dst = f"{R}/results/pilot/{sid}/{arm}", f"{R}/results/pilot/{sid}/{arm}_masked"; os.makedirs(dst, exist_ok=True); n = 0
        for i in range(8):
            if not os.path.exists(f"{src}/{i:03d}.png") or not os.path.exists(f"{R}/results/phaseA/{sid}/D_render/alpha/{i:03d}.png"): continue
            base = np.asarray(Image.open(f"{R}/results/pilot/{sid}/D/images/{i:03d}.png").convert("RGB"), np.float32)
            y = np.asarray(Image.open(f"{src}/{i:03d}.png").convert("RGB").resize(base.shape[1::-1], Image.BICUBIC), np.float32)
            m = (load_view(f"{R}/results/phaseA/{sid}/D_render", i)[1] > 0.5).astype(np.float32)[..., None]
            Image.fromarray((y * m + base * (1 - m)).clip(0, 255).astype(np.uint8)).save(f"{dst}/{i:03d}.png"); n += 1
        print(sid, arm, n, "views masked")
