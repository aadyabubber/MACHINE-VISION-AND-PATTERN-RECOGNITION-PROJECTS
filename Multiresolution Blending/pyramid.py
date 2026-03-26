import os
import cv2
import numpy as np
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

def to_float01(img_bgr_uint8: np.ndarray) -> np.ndarray:
    return img_bgr_uint8.astype(np.float32) / 255.0

def to_uint8(img_float01: np.ndarray) -> np.ndarray:
    img = np.clip(img_float01, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)

def gaussian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    G = [img]
    for _ in range(levels):
        img = cv2.pyrDown(img)
        G.append(img)
    return G

def laplacian_pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
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
    levels = 0
    mh, mw = h, w
    while levels < cap and mh >= 4 and mw >= 4:
        mh //= 2
        mw //= 2
        levels += 1
    return max(levels, 1)

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m <= 1e-12:
        return float("inf")
    return float(10.0 * math.log10(1.0 / m))

def tile_pyramid_for_display(pyr: list[np.ndarray], mode: str) -> np.ndarray:
    """
    mode:
      - 'gauss': expects values in [0,1]
      - 'laplace': can be negative; we normalize each level for display
    Returns a single uint8 BGR image that tiles levels horizontally.
    """
    tiles = []
    base_h = pyr[0].shape[0]

    for i, im in enumerate(pyr):
        x = im.copy()

        if mode == "gauss":
            # already [0,1]
            disp = to_uint8(x)
        elif mode == "laplace":
            # normalize each level for display so edges show up
            # For Laplacian, values are around [-?, +?]. Use per-level min/max.
            mn = float(np.min(x))
            mx = float(np.max(x))
            if abs(mx - mn) < 1e-12:
                disp = np.zeros_like(x, dtype=np.uint8)
            else:
                x01 = (x - mn) / (mx - mn)
                disp = to_uint8(x01)
        else:
            raise ValueError("Unknown mode")

        # resize each tile to have same height (base_h) while preserving aspect
        h, w = disp.shape[:2]
        new_w = int(round(w * (base_h / h)))
        disp_resized = cv2.resize(disp, (new_w, base_h), interpolation=cv2.INTER_AREA)

        # add a small label bar (optional)
        label = f"L{i}"
        cv2.putText(disp_resized, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        tiles.append(disp_resized)

    # concatenate horizontally
    return cv2.hconcat(tiles)

def save_pyramids_for_one_image(img_path: str, tag: str, cap_levels: int = 6):
    img_u8 = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_u8 is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    img = to_float01(img_u8)
    h, w = img.shape[:2]
    levels = max_pyramid_levels(h, w, cap=cap_levels)

    G = gaussian_pyramid(img, levels)
    L = laplacian_pyramid(img, levels)
    recon = collapse_laplacian_pyramid(L)

    # metrics
    m = mse(img, recon)
    p = psnr(img, recon)
    mxerr = float(np.max(np.abs(img - recon)))
    print(f"[{tag}] levels={levels} MSE={m:.8e} PSNR={p:.2f} max|err|={mxerr:.8e}")

    # visuals
    gauss_vis = tile_pyramid_for_display(G, mode="gauss")
    lap_vis = tile_pyramid_for_display(L, mode="laplace")

    # reconstruction + difference (amplify diff so it's visible)
    recon_u8 = to_uint8(recon)
    diff = np.abs(img - recon)  # [0,1] but tiny
    diff_amp = np.clip(diff * 5000.0, 0.0, 1.0)  # amplify
    diff_u8 = to_uint8(diff_amp)

    cv2.imwrite(os.path.join(OUTDIR, f"{tag}_gaussian_pyr.png"), gauss_vis)
    cv2.imwrite(os.path.join(OUTDIR, f"{tag}_laplacian_pyr.png"), lap_vis)
    cv2.imwrite(os.path.join(OUTDIR, f"{tag}_reconstruction.png"), recon_u8)
    cv2.imwrite(os.path.join(OUTDIR, f"{tag}_recon_absdiff_x5000.png"), diff_u8)

    print(f"Saved pyramid visuals to ./{OUTDIR}/ for tag='{tag}'")

def main():
    # Pick one image to visualize (desert is typical)
    save_pyramids_for_one_image("desert.png", tag="desert", cap_levels=6)

    # If you ALSO want sea, uncomment:
    # save_pyramids_for_one_image("sea.png", tag="sea", cap_levels=6)

if __name__ == "__main__":
    main()
