from PIL import Image, ImageDraw, ImageFont
import os

# Image paths
paths = {
    "original":   "/work/mech-ai-scratch/UshB/sage-extend/leaf_input/image.png",
    "mask":       "/work/mech-ai-scratch/UshB/sage-extend/leaf_input/0.png",
    "sam3d":      "/work/mech-ai-scratch/UshB/sage-extend/leaf_output/leaf_mesh.png",
    "gaussian":   "/work/mech-ai-scratch/UshB/sage-extend/Modified/leaf_anthracnose.png",
}

labels = [
    "Ground Truth\n(SAGE Input)",
    "SAM3D\nMask",
    "SAM3D\nOutput (3D)",
    "GaussianEditor\nOutput (3D)",
]

# Layout constants
THUMB = 320          # target height for each image
PAD   = 30           # padding between/around cells
LABEL_H = 70         # height reserved for label below image
HEADER_H = 60        # top title bar height
BG    = (245, 245, 245)
LABEL_BG = (255, 255, 255)
BORDER_3D = (52, 120, 200)   # blue border for 3D outputs
BORDER_GT = (80, 160, 80)    # green border for ground truth
BORDER_MASK = (160, 160, 160)

border_colors = [BORDER_GT, BORDER_MASK, BORDER_3D, BORDER_3D]

# Load and resize images to fixed height
imgs = []
for key in paths:
    im = Image.open(paths[key]).convert("RGBA")
    scale = THUMB / im.height
    im = im.resize((int(im.width * scale), THUMB), Image.LANCZOS)
    # paste on white if RGBA
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bg.paste(im, mask=im.split()[3])
    imgs.append(bg)

cell_widths = [im.width for im in imgs]
n = len(imgs)
total_w = sum(cell_widths) + PAD * (n + 1) + 40
total_h = HEADER_H + PAD + THUMB + LABEL_H + PAD

canvas = Image.new("RGB", (total_w, total_h), BG)
draw = ImageDraw.Draw(canvas)

# Try to load a font, fall back to default
try:
    font_label = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 18)
    font_title = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 22)
    font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 14)
except Exception:
    font_label = ImageFont.load_default()
    font_title = font_label
    font_small = font_label

# Title
title = "3D Diseased Leaf Generation Pipeline"
bbox = draw.textbbox((0, 0), title, font=font_title)
tw = bbox[2] - bbox[0]
draw.text(((total_w - tw) // 2, (HEADER_H - (bbox[3] - bbox[1])) // 2),
          title, fill=(30, 30, 30), font=font_title)

# Draw each image cell
x = PAD
for i, im in enumerate(imgs):
    y_img = HEADER_H + PAD
    bw = 3  # border width
    bc = border_colors[i]
    # border rect
    draw.rectangle([x - bw, y_img - bw, x + im.width + bw, y_img + THUMB + bw],
                   outline=bc, width=bw)
    canvas.paste(im, (x, y_img))

    # 3D badge
    if i >= 2:
        badge = " 3D Output "
        bb = draw.textbbox((0, 0), badge, font=font_small)
        bw2, bh2 = bb[2] - bb[0], bb[3] - bb[1]
        bx, by = x + im.width - bw2 - 6, y_img + 4
        draw.rectangle([bx - 4, by - 2, bx + bw2 + 4, by + bh2 + 2],
                       fill=BORDER_3D)
        draw.text((bx, by), badge, fill=(255, 255, 255), font=font_small)

    # Label
    y_lbl = y_img + THUMB + 8
    lbl = labels[i]
    lines = lbl.split("\n")
    for li, line in enumerate(lines):
        lb = draw.textbbox((0, 0), line, font=font_label)
        lw = lb[2] - lb[0]
        lx = x + (im.width - lw) // 2
        draw.text((lx, y_lbl + li * 22), line, fill=(30, 30, 30), font=font_label)

    x += im.width + PAD

# Footer note
note = "SAM3D and GaussianEditor both produce 3D outputs (3D Gaussian Splatting)"
nb = draw.textbbox((0, 0), note, font=font_small)
nw = nb[2] - nb[0]
draw.text(((total_w - nw) // 2, total_h - 18), note, fill=(100, 100, 100), font=font_small)

out_path = "/work/mech-ai-scratch/UshB/sage-extend/pipeline_comparison.png"
canvas.save(out_path, dpi=(150, 150))
print(f"Saved: {out_path}  ({canvas.width}x{canvas.height})")
