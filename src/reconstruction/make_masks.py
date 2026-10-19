"""SAM (ViT-H) masks for pilot specimens from annotator box prompts.

Output: data/inputs/<specimen>/image.png + 0.png (mask), the layout SAM 3D's
load_single_mask expects, plus mask_meta.json (SAM score, coverage).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from PIL import Image
from transformers import SamModel, SamProcessor

ROOT = Path(__file__).resolve().parents[2]
cfg = yaml.safe_load(open(ROOT / "experiments/configs/pilot.yaml"))
man = pd.read_csv(ROOT / "data/metadata/specimen_manifest.csv").set_index("specimen_id")
dev = "cuda"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(dev).eval()
proc = SamProcessor.from_pretrained("facebook/sam-vit-huge")
for sid in cfg["specimens"]:
    out = ROOT / "data/inputs" / sid.replace("/", "__")
    out.mkdir(parents=True, exist_ok=True)
    im = Image.open(ROOT / man.loc[sid, "path"]).convert("RGB")
    W, H = im.size
    x0, y0, x1, y1 = cfg["boxes"][sid]
    box = [[[x0 * W, y0 * H, x1 * W, y1 * H]]]
    inp = proc(im, input_boxes=box, return_tensors="pt").to(dev)
    with torch.no_grad():
        o = model(**inp, multimask_output=True)
    masks = proc.image_processor.post_process_masks(o.pred_masks.cpu(), inp["original_sizes"].cpu(),
                                                    inp["reshaped_input_sizes"].cpu())[0][0]
    scores = o.iou_scores[0, 0].cpu().numpy()
    k = int(scores.argmax()); m = masks[k].numpy().astype(np.uint8) * 255
    im.save(out / "image.png"); Image.fromarray(m).save(out / "0.png")
    meta = dict(specimen_id=sid, box=cfg["boxes"][sid], sam_score=float(scores[k]), coverage=float((m > 0).mean()),
                sam_model="facebook/sam-vit-huge", size=[W, H])
    json.dump(meta, open(out / "mask_meta.json", "w"), indent=1)
    print(sid, f"score={scores[k]:.3f} cov={meta['coverage']:.2f}", flush=True)
