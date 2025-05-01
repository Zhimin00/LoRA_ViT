import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from r_pca import R_pca
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# === METRICS ===
def contrast_to_noise_ratio(signal, background):
    mean_signal = np.mean(signal)
    mean_background = np.mean(background)
    std_background = np.std(background)
    return np.abs(mean_signal - mean_background) / std_background

def speckle_contrast(image):
    return np.std(image) / np.mean(image)

# === RPCA Denoising ===
def rpca_denoise(img):
    rpca = R_pca(img)
    rpca.lam = 0.2 / np.sqrt(max(img.shape))  # manually override λ for stronger denoising
    L, _ = rpca.fit(max_iter=300)
    return np.clip(L, 0, 1)  # Return normalized [0,1] denoised image

# === CONFIG ===
input_dir = r'/cis/net/r24a/data/zshao/CS/raw_test_images'
output_dir = r'/cis/net/r24a/data/zshao/CS/raw_test_denoised2'
os.makedirs(output_dir, exist_ok=True)

# === PROCESS ALL IMAGES ===
metrics = []

file_list = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

for fname in tqdm(file_list, desc="Running RPCA Denoising"):
    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname)  # Keep same filename

    # Load image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    img_norm = img.astype(np.float32) / 255.0

    # Apply RPCA
    L = rpca_denoise(img_norm)
    L_uint8 = (L * 255).astype(np.uint8)

    # Save denoised image
    cv2.imwrite(output_path, L_uint8)

    # Compute metrics
    roi = slice(100, 150), slice(100, 150)
    signal = slice(50, 100), slice(50, 100)
    background = slice(150, 200), slice(150, 200)

    psnr_val = psnr(img_norm, L)
    ssim_val = ssim(img_norm, L, data_range=1.0)
    speckle_val = speckle_contrast(L)
    speckle_raw = speckle_contrast(img_norm)
    cnr_val = contrast_to_noise_ratio(L[signal], L[background])

    metrics.append([fname, psnr_val, ssim_val, speckle_raw, speckle_val, cnr_val])

# === SAVE METRICS
metrics_df = pd.DataFrame(metrics, columns=["Filename", "PSNR", "SSIM", "Speckle_Contrast_raw", "Speckle_Contrast_denoise", "CNR"])
metrics_csv_path = os.path.join(output_dir, "rpca_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)

print(f"Done! Metrics saved at {metrics_csv_path}")