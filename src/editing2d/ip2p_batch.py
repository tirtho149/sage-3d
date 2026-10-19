"""Batch InstructPix2Pix: one model load, many jobs. Job JSON: list of
{"image":..., "mask":..., "prompt":..., "seed":int, "out":...}. Skips existing outputs."""
import json, sys, os, time, torch, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
jobs = json.load(open(sys.argv[1]))
import importlib.util
spec = importlib.util.spec_from_file_location("ip2p_mod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "ip2p_core.py"))
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
for j in jobs:
    if os.path.exists(j["out"]):
        continue
    os.makedirs(os.path.dirname(j["out"]), exist_ok=True)
    t = time.time()
    img = Image.open(j["image"]).convert("RGB")
    mask = np.asarray(Image.open(j["mask"]).convert("L")) if j.get("mask") else np.full(img.size[::-1], 255, np.uint8)
    core.edit(img, mask, j["prompt"], j["seed"], igs=j.get("igs", 1.5), dilate=j.get("dilate", 0.0)).save(j["out"])
    json.dump(dict(j, sec=round(time.time() - t, 1)), open(j["out"] + ".json", "w"), indent=1)
    print("IP2P", j["out"], round(time.time() - t, 1), flush=True)
