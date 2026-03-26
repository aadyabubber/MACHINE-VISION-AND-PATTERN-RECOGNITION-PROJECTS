# p4_solution.py
import os
import math
import cv2
import numpy as np

# ----------------------------
# Utilities
# ----------------------------
def to_float01(img_bgr_uint8: np.ndarray) -> np.ndarray:
    """BGR uint8 -> float32 in [0,1]."""
    return img_bgr_uint8.astype(np.float32) / 255.0

def to_uint8(img_float01: np.ndarray) -> np.ndarray:
    """float32 in [0,1] -> uint8 BGR."""
    img = np.clip(img_float01, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)

def ensure_same_size(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Resize img2 to img1 size if needed."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1, w1) != (h2, w2):
        img2 = cv2.resize(img2, (w1, h1), interpolation=cv2.INTER_AREA)
    return img1, img2

def make_lr_mask(h: int, w: int) -> np.ndarray:
    """
    Binary mask M: left side 0, right side 1.
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

# ----------------------------
# Gaussian / Laplacian pyramids (B)
# ----------------------------
def gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    G = [img]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        G.append(img)
    return G

def laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """
    Return Laplacian pyramid with 'levels' Laplacian layers + final Gaussian residual.
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
    Safe pyramid depth given image size.
    cap=6 is usually plenty; reduce if images are small.
    """
    levels = 0
    mh, mw = h, w
    while levels < cap and mh >= 2 and mw >= 2:
        mh //= 2
        mw //= 2
        if mh < 2 or mw < 2:
            break
        levels += 1
    return max(levels, 1)

# ----------------------------
# Blending (C, D, E, F)
# ----------------------------
def direct_blend(I1: np.ndarray, I2: np.ndarray, M: np.ndarray) -> np.ndarray:
    return (1.0 - M) * I1 + M * I2

def alpha_blend(I1: np.ndarray, I2: np.ndarray, M: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Feather blending: blur mask, then blend.
    """
    # Blur expects 2D; blur then add channel back
    M2 = M[..., 0]
    # ksize should be odd and reasonably related to sigma
    k = int(max(3, 6 * sigma + 1))
    if k % 2 == 0:
        k += 1
    Mb = cv2.GaussianBlur(M2, (k, k), sigmaX=sigma, sigmaY=sigma)
    Mb = Mb[..., None].astype(np.float32)
    return (1.0 - Mb) * I1 + Mb * I2

def multiblend(I1: np.ndarray, I2: np.ndarray, M: np.ndarray, levels: int) -> np.ndarray:
    """
    Multiresolution blending (Burt & Adelson style):
    - Gaussian pyramid of mask
    - Laplacian pyramids of images
    - blend each level, then collapse
    """
    # Build pyramids
    L1 = laplacian_pyramid(I1, levels)
    L2 = laplacian_pyramid(I2, levels)

    # Mask Gaussian pyramid (need same number of levels as Laplacian list)
    # Laplacian list length = levels+1 (includes residual)
    GM = gaussian_pyramid(M, levels)

    # Blend per level
    LS = []
    for k in range(levels + 1):
        Mk = GM[k]
        # Ensure mask has same spatial size as the Laplacian level
        if Mk.shape[:2] != L1[k].shape[:2]:
            Mk = cv2.resize(Mk, (L1[k].shape[1], L1[k].shape[0]), interpolation=cv2.INTER_LINEAR)
        # If mask is single-channel, broadcast to 3 channels
        if Mk.shape[2] == 1 and L1[k].shape[2] == 3:
            Mk = np.repeat(Mk, 3, axis=2)
        LS.append((1.0 - Mk) * L1[k] + Mk * L2[k])

    # Collapse blended Laplacian pyramid
    return collapse_laplacian_pyramid(LS)

# ----------------------------
# Run on your 3 pairs
# ----------------------------
def run_pair(name: str, path1: str, path2: str, outdir: str, alpha_sigma: float = 10.0, cap_levels: int = 6):
    I1_u8 = cv2.imread(path1, cv2.IMREAD_COLOR)
    I2_u8 = cv2.imread(path2, cv2.IMREAD_COLOR)
    if I1_u8 is None or I2_u8 is None:
        raise FileNotFoundError(f"Could not read {path1} or {path2}")

    # Convert to float and match sizes
    I1 = to_float01(I1_u8)
    I2 = to_float01(I2_u8)
    I1, I2 = ensure_same_size(I1, I2)

    h, w = I1.shape[:2]
    levels = max_pyramid_levels(h, w, cap=cap_levels)

    # (B) Verify Laplacian reconstruction for both images
    for tag, img in [("img1", I1), ("img2", I2)]:
        L = laplacian_pyramid(img, levels)
        recon = collapse_laplacian_pyramid(L)
        m = mse(img, recon)
        p = psnr(img, recon)
        mx = float(np.max(np.abs(img - recon)))
        print(f"[{name}] recon {tag}: levels={levels}  MSE={m:.8e}  PSNR={p:.2f} dB  max|err|={mx:.8e}")

    # (C) Binary mask
    M = make_lr_mask(h, w)  # shape (h,w,1)
    # Make 3-channel mask for direct/alpha blending convenience
    M3 = np.repeat(M, 3, axis=2)

    # (D) Direct
    direct = direct_blend(I1, I2, M3)

    # (E) Alpha/feather
    alpha = alpha_blend(I1, I2, M, sigma=alpha_sigma)  # uses 1-channel mask internally

    # (F) Multiresolution
    blended = multiblend(I1, I2, M, levels=levels)

    # Save
    os.makedirs(outdir, exist_ok=True)
    cv2.imwrite(os.path.join(outdir, f"{name}_direct.png"), to_uint8(direct))
    cv2.imwrite(os.path.join(outdir, f"{name}_alpha.png"), to_uint8(alpha))
    cv2.imwrite(os.path.join(outdir, f"{name}_multires.png"), to_uint8(blended))

    # Also save the mask (visual)
    cv2.imwrite(os.path.join(outdir, f"{name}_mask.png"), to_uint8(np.repeat(M, 3, axis=2)))

def main():
    outdir = "outputs"

    pairs = [
        ("desert_sea", "desert.png", "sea.png"),
        ("cat_dog", "cat_sunset.png", "dog_sunset.png"),
        ("forest_alpine", "forest.png", "alpine.png"),
    ]

    for name, p1, p2 in pairs:
        run_pair(name, p1, p2, outdir=outdir, alpha_sigma=10.0, cap_levels=6)

    print(f"\nDone. Saved results to ./{outdir}/")
    print("For each pair you have: *_direct.png, *_alpha.png, *_multires.png, *_mask.png")

if __name__ == "__main__":
    main()
