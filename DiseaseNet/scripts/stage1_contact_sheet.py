"""Stage 1 audit: per-class contact sheet of the Stage-0 arms for one parent (view 0 and view 4, condition R1).
Columns: PHOTO, GEN2D, KB2D, U3D v0, KB3D v0, U3D v4, KB3D v4. usage: python scripts/stage1_contact_sheet.py [asset_rank]
-> figures/stage1/contact_sheet_r<rank>.jpg"""
import os, sys
import pandas as pd
from PIL import Image, ImageDraw
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rank = int(sys.argv[1]) if len(sys.argv) > 1 else 0
d = pd.concat([pd.read_csv(f"{ROOT}/data/synthetic/stage0/{m}") for m in ("manifest_3d.csv", "manifest_2d.csv")])
cols = [("PHOTO", 0), ("GEN2D", 0), ("KB2D", 0), ("U3D", 0), ("KB3D", 0), ("U3D", 4), ("KB3D", 4)]
classes = sorted(d.class_id.unique()); S, H = 150, 14
sheet = Image.new("RGB", (S * len(cols) + 190, (S + 2) * len(classes) + H), "white"); dr = ImageDraw.Draw(sheet)
for j, (a, v) in enumerate(cols):
    dr.text((190 + j * S + 4, 1), f"{a} v{v}", fill="black")
for i, c in enumerate(classes):
    aid = sorted(d[d.class_id == c].asset_id.unique())[rank]
    dr.text((2, H + i * (S + 2) + S // 2 - 6), c.replace("::", "\n"), fill="black")
    for j, (a, v) in enumerate(cols):
        r = d[(d.arm == a) & (d.asset_id == aid) & (d.view == v) & (d.cond == "R1")]
        if len(r):
            im = Image.open(f"{ROOT}/{r.path.iloc[0]}").convert("RGB"); im.thumbnail((S, S))
            sheet.paste(im, (190 + j * S, H + i * (S + 2)))
sheet.save(f"{ROOT}/figures/stage1/contact_sheet_r{rank}.jpg", quality=88)
print(sheet.size)
