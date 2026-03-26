import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------------------------
# IMAGE PATHS
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
IMAGE1_PATH = os.path.join(script_dir, "market.jpeg")   # Image 1
IMAGE2_PATH = os.path.join(script_dir, "duck.jpg")    # Image 2

# --------------------------------------------------
# Read Image 1 (Market)
# --------------------------------------------------
img1_bgr = cv2.imread(IMAGE1_PATH)
if img1_bgr is None:
    raise FileNotFoundError("duck.jpeg not found")

img1_rgb = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB)

# --------------------------------------------------
# (b) Crop Image 1
# --------------------------------------------------
h1, w1 = img1_rgb.shape[:2]
x1, y1 = int(0.05 * w1), int(0.35 * h1)
x2, y2 = int(0.55 * w1), int(0.75 * h1)
crop = img1_rgb[y1:y2, x1:x2]

# --------------------------------------------------
# Read Image 2 (Duck)
# --------------------------------------------------
img2_bgr = cv2.imread(IMAGE2_PATH)
if img2_bgr is None:
    raise FileNotFoundError("duck.jpg not found")

img2_rgb = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2RGB)
h2, w2 = img2_rgb.shape[:2]

# --------------------------------------------------
# (c) Downsample by 10×
# --------------------------------------------------
down = cv2.resize(img2_rgb, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)

# --------------------------------------------------
# (d) Upsample back
# --------------------------------------------------
up_nearest = cv2.resize(down, (w2, h2), interpolation=cv2.INTER_NEAREST)
up_bicubic = cv2.resize(down, (w2, h2), interpolation=cv2.INTER_CUBIC)

# --------------------------------------------------
# (e) Difference images
# --------------------------------------------------
diff_nearest = cv2.absdiff(img2_rgb, up_nearest)
diff_bicubic = cv2.absdiff(img2_rgb, up_bicubic)

# ==================================================
# PLOTTING SEPARATED BY PROBLEM 1 PARTS
# ==================================================

# -------------------------
# Problem 1(a): BGR vs RGB
# -------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img1_bgr)  # wrong colors on purpose
plt.title("Problem 1(a): Market – Wrong (BGR)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img1_rgb)
plt.title("Problem 1(a): Market – Correct (RGB)")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "output_1a_bgr_vs_rgb.png"), dpi=150, bbox_inches='tight')
plt.show()

# -------------------------
# Problem 1(b): Crop
# -------------------------
plt.figure(figsize=(6, 5))
plt.imshow(crop)
plt.title("Problem 1(b): Market – Cropped Region")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "output_1b_crop.png"), dpi=150, bbox_inches='tight')
plt.show()

# -------------------------
# Problem 1(c): Downsample
# -------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img2_rgb)
plt.title("Problem 1(c): Original Duck Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(down)
plt.title("Problem 1(c): Downsampled (10×)")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "output_1c_downsample.png"), dpi=150, bbox_inches='tight')
plt.show()

# -------------------------
# Problem 1(d): Upsample
# -------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(up_nearest)
plt.title("Problem 1(d): Upsampled – Nearest Neighbor")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(up_bicubic)
plt.title("Problem 1(d): Upsampled – Bicubic")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "output_1d_upsample.png"), dpi=150, bbox_inches='tight')
plt.show()

# -------------------------
# Problem 1(e): Differences
# -------------------------
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(diff_nearest)
plt.title("Problem 1(e): Difference – Nearest")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(diff_bicubic)
plt.title("Problem 1(e): Difference – Bicubic")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "output_1e_differences.png"), dpi=150, bbox_inches='tight')
plt.show()

# --------------------------------------------------
# Numerical error comparison
# --------------------------------------------------
print("(e) Sum of absolute pixel differences:")
print("Nearest Neighbor:", int(np.sum(diff_nearest)))
print("Bicubic:", int(np.sum(diff_bicubic)))
