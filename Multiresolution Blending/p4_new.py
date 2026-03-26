# p4_new.py
import os
import math
import cv2
import numpy as np

# ------------------------------------------------------------
# Always run relative to this script's directory (P4 folder)
# ------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

print("Script directory (P4):", SCRIPT_DIR)
print("Working directory set to:", os.getcwd())
print("Files here:", os.listdir("."))

# ------------------------------------------------------------
# Helper: load image with optional extensions
# ------------------------------------------------------------
def imread_flexible(stem_or_name: str):
    """
    Try reading:
      - exact name
      - name + .png / .jpg / .jpeg
    Returns BGR uint8 or None.
    """
    candidates = [
        stem_or_name,
        stem_or_name + ".png",
        stem_or_name + ".jpg",
        stem_or_name + ".jpeg",
    ]
    for c in candidates:
        img = cv2.imread(c, cv2.IMREAD_COLOR)
        if img is not None:
            return img, c
    return None, None

def to_float01(img_bgr_uint8: np.ndarray) -> np.ndarray:
    return img_bgr_uint8.astype(np.float32) / 255.0

def to_uint8(img_float01: np.ndarray) -> np.ndarray:
    img = np.clip(img_float01, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)

def ensure_same_size(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1, w1) != (h2, w2):
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)
    return img1, img2

def make_lr_mask(h: int, w: int) -> np.ndarray:
    """
    Binary mask:
      left half = 0, right half = 1
    Shape: (h,w,1) float32
    """
    M = np.zeros((h, w), dtype=np.float32)
    M[:, w // 2 :] = 1.0
    return M[..., None]

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / m))

# ------------------------------------------------------------
# B) Gaussian / Laplacian pyramids + reconstruction
# ------------------------------------------------------------
def gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    G = [img]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        G.append(img)
    return G

def laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """
    Laplacian pyramid: [L0, L1, ..., L_{levels-1}, residual]
    length = levels + 1
    """
    G = gaussian_pyramid(img, levels)
    L = []
    for i in range(levels):
        up = cv2.pyrUp(G[i + 1], dstsize=(G[i].shape[1], G[i].shape[0]))
        L.append(G[i] - up)
    L.append(G[-1])  # residual
    return L

def collapse_laplacian_pyramid(L: list[np.ndarray]) -> np.ndarray:
    img = L[-1]
    for i in range(len(L) - 2, -1, -1):
        img = cv2.pyrUp(img, dstsize=(L[i].shape[1], L[i].shape[0]))
        img = img + L[i]
    return img

def max_pyramid_levels(h: int, w: int, cap: int = 6) -> int:
    """
    Choose a safe number of pyramid levels for this image size.
    """
    levels = 0
    mh, mw = h, w
    while levels < cap and mh >= 4 and mw >= 4:
        mh //= 2
        mw //= 2
        levels += 1
    return max(levels, 1)

# ------------------------------------------------------------
# C/D/E/F) Blending
# ------------------------------------------------------------
def direct_blend(I1: np.ndarray, I2: np.ndarray, M3: np.ndarray) -> np.ndarray:
    return (1.0 - M3) * I1 + M3 * I2

def alpha_blend(I1: np.ndarray, I2: np.ndarray, M1: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Feathering: blur mask then blend.
    M1 is (h,w,1) float
    """
    M2 = M1[..., 0]
    k = int(max(3, 6 * sigma + 1))
    if k % 2 == 0:
        k += 1
    Mb = cv2.GaussianBlur(M2, (k, k), sigmaX=sigma, sigmaY=sigma)
    Mb = Mb[..., None].astype(np.float32)
    Mb3 = np.repeat(Mb, 3, axis=2)
    return (1.0 - Mb3) * I1 + Mb3 * I2

def multiblend(I1: np.ndarray, I2: np.ndarray, M1: np.ndarray, levels: int) -> np.ndarray:
    """
    Multiresolution blending:
      - Gaussian pyramid of mask
      - Laplacian pyramids of images
      - Blend each level, collapse
    """
    L1 = laplacian_pyramid(I1, levels)
    L2 = laplacian_pyramid(I2, levels)
    GM = gaussian_pyramid(M1, levels)  # mask pyramid

    LS = []
    for k in range(levels + 1):
        Mk = GM[k]

        # --- FIX: ensure mask is always (h,w,1) ---
        if Mk.ndim == 2:
            Mk = Mk[..., None]
        elif Mk.ndim == 3 and Mk.shape[2] != 1:
            Mk = Mk[:, :, :1]

        # Ensure mask matches Laplacian size
        if Mk.shape[:2] != L1[k].shape[:2]:
            Mk = cv2.resize(Mk, (L1[k].shape[1], L1[k].shape[0]), interpolation=cv2.INTER_LINEAR)
            if Mk.ndim == 2:
                Mk = Mk[..., None]

        Mk3 = np.repeat(Mk, 3, axis=2)
        LS.append((1.0 - Mk3) * L1[k] + Mk3 * L2[k])

    return collapse_laplacian_pyramid(LS)


# ------------------------------------------------------------
# Run a pair and save outputs
# ------------------------------------------------------------
def run_pair(name: str, file1: str, file2: str, outdir: str, alpha_sigma: float = 10.0, cap_levels: int = 6):
    I1_u8, used1 = imread_flexible(file1)
    I2_u8, used2 = imread_flexible(file2)

    if I1_u8 is None or I2_u8 is None:
        raise FileNotFoundError(
            f"Could not read images for '{name}'.\n"
            f"Tried '{file1}' and '{file2}' with .png/.jpg/.jpeg in:\n{SCRIPT_DIR}"
        )

    print(f"\n[{name}] loaded: {used1} and {used2}")

    I1 = to_float01(I1_u8)
    I2 = to_float01(I2_u8)
    I1, I2 = ensure_same_size(I1, I2)

    h, w = I1.shape[:2]
    levels = max_pyramid_levels(h, w, cap=cap_levels)

    # (B) Verify reconstruction for both images
    for tag, img in [("img1", I1), ("img2", I2)]:
        L = laplacian_pyramid(img, levels)
        recon = collapse_laplacian_pyramid(L)
        m = mse(img, recon)
        p = psnr(img, recon)
        mx = float(np.max(np.abs(img - recon)))
        print(f"[{name}] recon {tag}: levels={levels}  MSE={m:.8e}  PSNR={p:.2f} dB  max|err|={mx:.8e}")

    # (C) mask
    M1 = make_lr_mask(h, w)                 # (h,w,1)
    M3 = np.repeat(M1, 3, axis=2)           # (h,w,3)

    # (D) direct
    direct = direct_blend(I1, I2, M3)

    # (E) alpha
    alpha = alpha_blend(I1, I2, M1, sigma=alpha_sigma)

    # (F) multires
    blended = multiblend(I1, I2, M1, levels=levels)

    # Save
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, f"{name}_mask.png"), to_uint8(M3))
    cv2.imwrite(os.path.join(outdir, f"{name}_direct.png"), to_uint8(direct))
    cv2.imwrite(os.path.join(outdir, f"{name}_alpha.png"), to_uint8(alpha))
    cv2.imwrite(os.path.join(outdir, f"{name}_multires.png"), to_uint8(blended))

    print(f"[{name}] saved: {name}_mask/direct/alpha/multires.png -> ./{outdir}/")

def main():
    outdir = "outputs"

    # Use stems (works whether your files are "desert" or "desert.png")
    pairs = [
    ("desert_sea", "desert.png", "sea.png"),
    ("cat_dog", "cat_sunset.png", "dog_sunset.png"),
    ("forest_alpine", "forest.png", "alpine.png"),
    ]


    for name, f1, f2 in pairs:
        run_pair(name, f1, f2, outdir=outdir, alpha_sigma=10.0, cap_levels=6)

    print("\nDONE Check the outputs/ folder for results.")

if __name__ == "__main__":
    main()
