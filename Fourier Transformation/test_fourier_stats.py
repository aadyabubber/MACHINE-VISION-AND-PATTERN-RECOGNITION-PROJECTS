import cv2
import numpy as np
import os

# Use script's directory
base = os.path.dirname(os.path.abspath(__file__))
img1_path = os.path.join(base, "happy.jpg")
img2_path = os.path.join(base, "sad.jpg")

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

print("\n=== FOURIER MAGNITUDE STATISTICS ===\n")
print(f"Image 1 (happy.jpg) shape: {img1.shape}")
print(f"Image 2 (sad.jpg) shape: {img2.shape}")

# Fourier transform
F1 = np.fft.fftshift(np.fft.fft2(img1))
F2 = np.fft.fftshift(np.fft.fft2(img2))

mag1 = np.log(np.abs(F1) + 1e-8)
mag2 = np.log(np.abs(F2) + 1e-8)

print("\n--- BEFORE FILTERING ---")
print(f"Original Magnitude - Min: {np.min(mag1):.6f}, Max: {np.max(mag1):.6f}")
print(f"Original Magnitude - Mean: {np.mean(mag1):.6f}, Std: {np.std(mag1):.6f}")

# Create filters
def low_pass_mask(shape, radius):
    H, W = shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    return (((X - cx) ** 2 + (Y - cy) ** 2) <= radius ** 2).astype(np.float32)

def high_pass_mask(shape, radius):
    return 1.0 - low_pass_mask(shape, radius)

def diagonal_bandpass_mask(shape, band_width=10):
    H, W = shape
    Y, X = np.ogrid[:H, :W]
    return (np.abs((Y - H // 2) - (X - W // 2)) < band_width).astype(np.float32)

# Apply filters
lp_mask = low_pass_mask(img1.shape, radius=40)
hp_mask = high_pass_mask(img1.shape, radius=40)
db_mask = diagonal_bandpass_mask(img1.shape, band_width=8)

F_lp = F1 * lp_mask
F_hp = F1 * hp_mask
F_db = F1 * db_mask

mag_lp = np.log(np.abs(F_lp) + 1e-8)
mag_hp = np.log(np.abs(F_hp) + 1e-8)
mag_db = np.log(np.abs(F_db) + 1e-8)

print("\n--- AFTER FILTERING ---")
print(f"Low-pass Magnitude   - Min: {np.min(mag_lp):.6f}, Max: {np.max(mag_lp):.6f}")
print(f"Low-pass Magnitude   - Mean: {np.mean(mag_lp):.6f}, Std: {np.std(mag_lp):.6f}")

print(f"\nHigh-pass Magnitude  - Min: {np.min(mag_hp):.6f}, Max: {np.max(mag_hp):.6f}")
print(f"High-pass Magnitude  - Mean: {np.mean(mag_hp):.6f}, Std: {np.std(mag_hp):.6f}")

print(f"\nDiagonal BP Magnitude - Min: {np.min(mag_db):.6f}, Max: {np.max(mag_db):.6f}")
print(f"Diagonal BP Magnitude - Mean: {np.mean(mag_db):.6f}, Std: {np.std(mag_db):.6f}")

print("\n=== COMPLETED SUCCESSFULLY ===")
