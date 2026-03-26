import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Disable interactive plots - just print to terminal
plt.ioff()
# Or use non-interactive backend
# matplotlib.use('Agg')

# ===============================
# Utility functions
# ===============================
def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img.astype(np.float32) / 255.0

def fft2_image(img):
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    mag = np.log(np.abs(Fshift) + 1e-8)
    phase = np.angle(Fshift)
    return Fshift, mag, phase

def ifft2_image(Fshift):
    F = np.fft.ifftshift(Fshift)
    img = np.fft.ifft2(F)
    return np.real(img)

def show_fourier(mag, phase, title):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(mag, cmap='gray')
    plt.title(f"{title} – Magnitude (log)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(phase, cmap='gray')
    plt.title(f"{title} – Phase")
    plt.axis('off')
    plt.close()

def show_magnitude_only(mag, title):
    plt.figure(figsize=(5, 4))
    plt.imshow(mag, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.close()

# ===============================
# (b) Frequency filters
# ===============================
def low_pass_mask(shape, radius):
    H, W = shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    return (((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2).astype(np.float32)

def high_pass_mask(shape, radius):
    return 1.0 - low_pass_mask(shape, radius)

def diagonal_bandpass_mask(shape, band_width=10):
    """
    Pass frequencies close to the main diagonal line (u ~ v) in the shifted spectrum.
    band_width controls thickness of the diagonal band.
    """
    H, W = shape
    Y, X = np.ogrid[:H, :W]
    return (np.abs((Y - H // 2) - (X - W // 2)) < band_width).astype(np.float32)

# ===============================
# (c) Phase swapping
# ===============================
def combine_mag_phase(F_mag_source, F_phase_source):
    mag = np.abs(F_mag_source)
    phase = np.angle(F_phase_source)
    return mag * np.exp(1j * phase)

# ===============================
# (d) Hybrid image (Oliva et al.)
# ===============================
def gaussian_lowpass_mask(shape, sigma):
    """
    Gaussian low-pass in frequency domain (centered).
    sigma is in frequency-domain pixels (bigger => less aggressive LP).
    """
    H, W = shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    D2 = (X - cx) ** 2 + (Y - cy) ** 2
    return np.exp(-D2 / (2 * sigma ** 2)).astype(np.float32)

# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    # Use script's directory (put happy.jpg and sad.jpg in same folder as this script)
    base = os.path.dirname(os.path.abspath(__file__))
    img1_path = os.path.join(base, "happy.jpg")
    img2_path = os.path.join(base, "sad.jpg")

    img1 = read_gray(img1_path)
    img2 = read_gray(img2_path)

    # ---------------------------
    # (a) Fourier transform
    # ---------------------------
    print("\n=== PART (a): FOURIER TRANSFORM ===")
    print(f"Image 1 (happy.jpg) - Grayscale shape: {img1.shape}, Range: [{np.min(img1):.4f}, {np.max(img1):.4f}]")
    print(f"Image 2 (sad.jpg) - Grayscale shape: {img2.shape}, Range: [{np.min(img2):.4f}, {np.max(img2):.4f}]")
    
    # Save the original grayscale image
    base_out = os.path.dirname(os.path.abspath(__file__))
    img1_gray_uint8 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(base_out, "img1_grayscale.png"), img1_gray_uint8)
    print(f"Saved: img1_grayscale.png")
    
    # Compute Fourier transform
    F1, mag1, phase1 = fft2_image(img1)
    
    # Save magnitude and phase as images
    mag1_uint8 = (np.clip((mag1 - np.min(mag1)) / (np.max(mag1) - np.min(mag1) + 1e-8), 0, 1) * 255).astype(np.uint8)
    phase1_uint8 = (np.clip((phase1 - np.min(phase1)) / (np.max(phase1) - np.min(phase1) + 1e-8), 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(base_out, "fft_magnitude.png"), mag1_uint8)
    cv2.imwrite(os.path.join(base_out, "fft_phase.png"), phase1_uint8)
    print(f"Saved: fft_magnitude.png, fft_phase.png")
    
    show_fourier(mag1, phase1, "Image 1 (happy)")

    # ---------------------------
    # (b) Frequency-domain filtering WITH Fourier magnitude plots
    # ---------------------------
    # Print original Fourier magnitude statistics
    print("\n=== FOURIER MAGNITUDE STATISTICS (BEFORE FILTERING) ===")
    print(f"Original Magnitude - Min: {np.min(mag1):.6f}, Max: {np.max(mag1):.6f}, Mean: {np.mean(mag1):.6f}, Std: {np.std(mag1):.6f}")
    
    # Plot original Fourier magnitude (required by the prompt for part b)
    plt.figure(figsize=(6, 5))
    plt.imshow(mag1, cmap='gray')
    plt.title("BEFORE FILTERING: Original Fourier Magnitude")
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(base_out, "b_magnitude_before_filtering.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: b_magnitude_before_filtering.png")

    # Create filters
    lp_mask = low_pass_mask(img1.shape, radius=40)
    hp_mask = high_pass_mask(img1.shape, radius=40)
    db_mask = diagonal_bandpass_mask(img1.shape, band_width=8)

    # Apply filters in frequency domain
    F_lp = F1 * lp_mask
    F_hp = F1 * hp_mask
    F_db = F1 * db_mask

    # Print filtered Fourier magnitude statistics
    print("\n=== FOURIER MAGNITUDE STATISTICS (AFTER FILTERING) ===")
    mag_lp = np.log(np.abs(F_lp) + 1e-8)
    mag_hp = np.log(np.abs(F_hp) + 1e-8)
    mag_db = np.log(np.abs(F_db) + 1e-8)
    print(f"Low-pass Magnitude   - Min: {np.min(mag_lp):.6f}, Max: {np.max(mag_lp):.6f}, Mean: {np.mean(mag_lp):.6f}, Std: {np.std(mag_lp):.6f}")
    print(f"High-pass Magnitude  - Min: {np.min(mag_hp):.6f}, Max: {np.max(mag_hp):.6f}, Mean: {np.mean(mag_hp):.6f}, Std: {np.std(mag_hp):.6f}")
    print(f"Diagonal BP Magnitude - Min: {np.min(mag_db):.6f}, Max: {np.max(mag_db):.6f}, Mean: {np.mean(mag_db):.6f}, Std: {np.std(mag_db):.6f}")

    # Plot Fourier magnitudes after filtering (required)
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(mag_lp, cmap='gray')
    plt.title("Low-pass Filtered Magnitude")
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mag_hp, cmap='gray')
    plt.title("High-pass Filtered Magnitude")
    plt.colorbar()
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mag_db, cmap='gray')
    plt.title("Diagonal Band-pass Filtered Magnitude")
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(base_out, "b_magnitude_after_filtering.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: b_magnitude_after_filtering.png")

    # Inverse FFT to get filtered images
    img_lp = ifft2_image(F_lp)
    img_hp = ifft2_image(F_hp)
    img_db = ifft2_image(F_db)

    # Display filtered images
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_lp, cmap='gray')
    plt.title("Low-pass Filtered Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_hp, cmap='gray')
    plt.title("High-pass Filtered Image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_db, cmap='gray')
    plt.title("Diagonal Band-pass Filtered Image")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(base_out, "b_filtered_images.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: b_filtered_images.png")

    # ---------------------------
    # (c) Phase swapping
    # ---------------------------
    F2, mag2, phase2 = fft2_image(img2)
    show_fourier(mag2, phase2, "Image 2 (sad)")

    # Swap phase
    F_img1_mag_img2_phase = combine_mag_phase(F1, F2)
    F_img2_mag_img1_phase = combine_mag_phase(F2, F1)

    img1_with_img2_phase = ifft2_image(F_img1_mag_img2_phase)
    img2_with_img1_phase = ifft2_image(F_img2_mag_img1_phase)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img1_with_img2_phase, cmap='gray')
    plt.title("Mag(img1) + Phase(img2)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img2_with_img1_phase, cmap='gray')
    plt.title("Mag(img2) + Phase(img1)")
    plt.axis('off')
    plt.close()

    # Modify phase differently than just swapping: partial phase blending
    alpha = 0.5
    blended_phase = alpha * phase1 + (1.0 - alpha) * phase2
    F_partial_phase = np.abs(F1) * np.exp(1j * blended_phase)
    img_partial_phase = ifft2_image(F_partial_phase)

    plt.figure(figsize=(5, 4))
    plt.imshow(img_partial_phase, cmap='gray')
    plt.title("Partial Phase Blending (α=0.5)")
    plt.axis('off')
    plt.close()

    # ---------------------------
    # (d) Hybrid image (Oliva et al.)
    # ---------------------------
    # NOTE: Hybrid images work best when faces are aligned (eyes/nose/mouth line up)
    sigma = 20  # tune this (try 10..40)
    LP = gaussian_lowpass_mask(img1.shape, sigma)
    HP = 1.0 - LP

    # Common practice:
    # - low frequencies from img1 (happy)
    # - high frequencies from img2 (sad)
    hybrid_F = F1 * LP + F2 * HP
    hybrid = ifft2_image(hybrid_F)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title("Low-frequency source (img1)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title("High-frequency source (img2)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(hybrid, cmap='gray')
    plt.title("Hybrid image")
    plt.axis('off')
    plt.close()
    
    print("\n=== ALL PLOTS PROCESSED SUCCESSFULLY ===")
