"""Generate a leaf mask from the diseased leaf image using HSV thresholding."""
import numpy as np
from PIL import Image
import cv2
import os

img_path = "/work/mech-ai-scratch/UshB/sage-extend/Screenshot 2026-10-04 at 11.09.07 PM.png"
out_dir = "/work/mech-ai-scratch/UshB/sage-extend/leaf_input"
os.makedirs(out_dir, exist_ok=True)

img = Image.open(img_path).convert("RGB")
img_np = np.array(img)

# Convert to HSV for robust green vs blue-gray separation
hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

# Leaf: hue 25-90 (yellow-green to green), reasonable saturation and value
# Background: hue ~100-130 (blue-gray), low saturation
leaf_mask = (
    (hsv[..., 0] >= 20) & (hsv[..., 0] <= 95) &
    (hsv[..., 1] >= 30) &
    (hsv[..., 2] >= 30)
)

# Also include the dark brown lesion spots (low saturation, dark)
# These are part of the leaf disease - include by checking they're not the blue-gray bg
not_bg = ~(
    (hsv[..., 0] >= 95) & (hsv[..., 0] <= 140) &
    (hsv[..., 1] < 60)
)
# Dark regions that aren't background (lesions)
dark_not_bg = (hsv[..., 2] < 80) & not_bg & (img_np[..., 0] < 100)

combined = leaf_mask | dark_not_bg

# Morphological cleanup: close small holes, remove tiny islands
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
combined_uint8 = combined.astype(np.uint8) * 255
combined_uint8 = cv2.morphologyEx(combined_uint8, cv2.MORPH_CLOSE, kernel, iterations=3)
combined_uint8 = cv2.morphologyEx(combined_uint8, cv2.MORPH_OPEN, kernel, iterations=1)

# Keep only the largest connected component
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_uint8)
largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
final_mask = (labels == largest).astype(np.uint8) * 255

# Save image (RGB PNG) and mask (grayscale PNG named 0.png)
img.save(os.path.join(out_dir, "image.png"))
Image.fromarray(final_mask).save(os.path.join(out_dir, "0.png"))

print(f"Saved image to {out_dir}/image.png")
print(f"Saved mask to {out_dir}/0.png")
print(f"Mask coverage: {final_mask.mean()/255*100:.1f}% of image")
