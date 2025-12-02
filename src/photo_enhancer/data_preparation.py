"""
This module provides functions for creating realistic, degraded images for training
image restoration models. It is the core of the data preparation pipeline.

The primary function, `apply_realesrgan_degradation`, implements a sophisticated
two-cycle degradation process inspired by the Real-ESRGAN paper. This process
simulates real-world image artifacts more accurately than simple downsampling,
leading to more robust and effective restoration models.
"""
import cv2
import random
import numpy as np
from scipy.special import i0
from typing import Dict, Any

# ==============================================================================
# REAL-ESRGAN DEGRADATION CONFIGURATION
# ==============================================================================
# This dictionary holds all parameters for the degradation pipeline.
# The pipeline consists of two degradation cycles, allowing for a more complex
# and realistic simulation of image artifacts.
realesrgan_degradation: Dict[str, Any] = {
    # --- First Degradation Cycle (More severe) ---
    "blur_prob1": 0.6,          # Probability of applying Gaussian blur.
    "kernel_range1": [3, 5, 7, 9], # Kernel sizes for Gaussian blur.
    "resize_prob1": 0.6,        # Probability of resizing the image.
    "resize_range1": [0.15, 3], # Scale factor range for resizing.
    "noise_prob1": 0.0,           # Probability of adding noise.
    "noise_sigma_range1": [0, 5], # Sigma range for Gaussian noise.
    "poisson_scale_range1": [0.05, 0.5], # Scale range for Poisson noise.
    "jpeg_prob1": 0.6,          # Probability of applying JPEG compression.
    "jpeg_quality_range1": [20, 70], # Quality range for JPEG compression.

    # --- Second Degradation Cycle (Milder) ---
    "blur_prob2": 0.3,
    "kernel_range2": [1, 3, 5, 7],
    "resize_prob2": 0.5,
    "resize_range2": [0.5, 1.5],
    "noise_prob2": 0.2,
    "noise_sigma_range2": [0, 0.5],
    "poisson_scale_range2": [0.001, 0.01],
    "jpeg_prob2": 0.6,
    "jpeg_quality_range2": [30, 90],

    # --- Final Processing Stage ---
    "sinc_prob": 0.3,           # Probability of applying a sinc filter to simulate ringing artifacts.
    "sinc_kernel_size": 5,      # Kernel size for the sinc filter.
    "final_jpeg_prob": 0.3,     # Probability of a final JPEG compression pass.
    "final_jpeg_quality_range": [30, 95]
}
# ==============================================================================

def _circular_lowpass_kernel(cutoff: float, kernel_size: int, pad_to: int = 0) -> np.ndarray:
    """
    Generates a 2D circular low-pass filter kernel for sinc filtering.

    This kernel is used to simulate ringing artifacts often seen in real-world
    digital images.

    Args:
        cutoff (float): The cutoff frequency of the filter.
        kernel_size (int): The size of the kernel (e.g., 21).
        pad_to (int): The size to pad the kernel to. Defaults to 0 (no padding).

    Returns:
        np.ndarray: The generated circular low-pass kernel.
    """
    kernel = np.fromfunction(
        lambda x, y: cutoff * i0(np.sqrt(
            (x - (kernel_size - 1) / 2)**2 + (y - (kernel_size - 1) / 2)**2) * cutoff),
        [kernel_size, kernel_size])
    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = 1
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel

def apply_realesrgan_degradation(hr_image: np.ndarray, scale: int = 1) -> np.ndarray:
    """
    Applies a two-cycle degradation process to a high-resolution image.

    This function simulates a variety of real-world image artifacts by applying
    a randomized sequence of degradations (blur, resize, noise, JPEG) in two
    separate cycles, followed by a final processing stage. This creates a
    realistic low-resolution counterpart for training robust image restoration models.

    Args:
        hr_image (np.ndarray): The input high-resolution image as a NumPy array
            in BGR format (as loaded by OpenCV).
        scale (int): The final downscaling factor to produce the low-resolution
            image. Defaults to 1.

    Returns:
        np.ndarray: The degraded low-resolution image as a NumPy array of type uint8.
    """
    cfg = realesrgan_degradation
    img = hr_image.astype(np.float32)
    h, w, _ = img.shape

    # --- First Degradation Cycle ---
    ops = random.sample(['blur', 'resize', 'noise', 'jpeg'], k=4)
    for op in ops:
        if op == 'blur' and random.random() < cfg["blur_prob1"]:
            kernel_size = random.choice(cfg["kernel_range1"])
            kernel = cv2.getGaussianKernel(kernel_size, 0)
            img = cv2.filter2D(img, -1, np.dot(kernel, kernel.transpose()))
        elif op == 'resize' and random.random() < cfg["resize_prob1"]:
            scale_factor = random.uniform(cfg['resize_range1'][0], cfg["resize_range1"][1])
            ch, cw, _ = img.shape
            img = cv2.resize(img, (int(cw * scale_factor), int(ch * scale_factor)), interpolation=cv2.INTER_LINEAR)
        elif op == 'noise' and random.random() < cfg["noise_prob1"]:
            if random.random() < 0.9: # Gaussian noise
                noise = np.random.normal(0, random.uniform(cfg["noise_sigma_range1"][0], cfg["noise_sigma_range1"][1]), img.shape)
                img += noise
            else: # Poisson noise
                poisson_scale = random.uniform(cfg["poisson_scale_range1"][0], cfg["poisson_scale_range1"][1])
                img_0_1 = np.clip(img, 0, 255) / 255.0
                noisy_img = np.random.poisson(img_0_1 * poisson_scale) / poisson_scale
                img = noisy_img * 255.0
        elif op == 'jpeg' and random.random() < cfg["jpeg_prob1"]:
            quality = random.randint(cfg["jpeg_quality_range1"][0], cfg["jpeg_quality_range1"][1])
            _, encimg = cv2.imencode('.jpg', np.clip(img, 0, 255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            img = cv2.imdecode(encimg, cv2.IMREAD_COLOR).astype(np.float32)

    # --- Second Degradation Cycle ---
    ops_2 = random.sample(['blur', 'resize', 'noise', 'jpeg'], k=4)
    for op in ops_2:
        if op == 'blur' and random.random() < cfg["blur_prob2"]:
            kernel_size = random.choice(cfg["kernel_range2"])
            kernel = cv2.getGaussianKernel(kernel_size, 0)
            img = cv2.filter2D(img, -1, np.dot(kernel, kernel.transpose()))
        elif op == 'resize' and random.random() < cfg["resize_prob2"]:
            scale_factor = random.uniform(cfg["resize_range2"][0], cfg["resize_range2"][1])
            ch, cw, _ = img.shape
            img = cv2.resize(img, (int(cw * scale_factor), int(ch * scale_factor)), interpolation=cv2.INTER_LINEAR)
        elif op == 'noise' and random.random() < cfg['noise_prob2']:
            if random.random() < 0.8: # Gaussian noise
                noise = np.random.normal(0, random.uniform(cfg["noise_sigma_range2"][0], cfg["noise_sigma_range2"][1]), img.shape)
                img += noise
            else: # Poisson noise
                poisson_scale = random.uniform(cfg["poisson_scale_range2"][0], cfg["poisson_scale_range2"][1])
                img_0_1 = np.clip(img, 0, 255) / 255.0
                noisy_img = np.random.poisson(img_0_1 * poisson_scale) / poisson_scale
                img = noisy_img * 255.0
        elif op == 'jpeg' and random.random() < cfg["jpeg_prob2"]:
            quality = random.randint(cfg["jpeg_quality_range2"][0], cfg["jpeg_quality_range2"][1])
            _, encimg = cv2.imencode('.jpg', np.clip(img, 0, 255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            img = cv2.imdecode(encimg, cv2.IMREAD_COLOR).astype(np.float32)

    # --- Final Processing ---
    # Resize to final target LR size
    lr_image = cv2.resize(np.clip(img, 0, 255), (int(w / scale), int(h / scale)), interpolation=cv2.INTER_LINEAR)

    # Apply sinc filter
    if random.random() < cfg["sinc_prob"]:
        sinc_kernel = _circular_lowpass_kernel(random.uniform(np.pi / 4, np.pi), cfg["sinc_kernel_size"], pad_to=False)
        lr_image = cv2.filter2D(lr_image, -1, sinc_kernel)

    # Final JPEG compression
    if random.random() < cfg["final_jpeg_prob"]:
        quality = random.randint(cfg["final_jpeg_quality_range"][0], cfg["final_jpeg_quality_range"][1])
        _, encimg = cv2.imencode('.jpg', np.clip(lr_image, 0, 255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        lr_image = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

    return np.clip(lr_image, 0, 255).astype(np.uint8)

def get_image_sharpness(image: np.ndarray) -> float:
    """
    Calculates image sharpness using the variance of the Sobel operator.

    A higher variance corresponds to more pronounced edges, indicating a sharper image.
    This is used to filter out blurry source images before they enter the
    degradation pipeline.

    Args:
        image (np.ndarray): The input image (BGR or grayscale).

    Returns:
        float: The calculated sharpness score.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.var(sobelx) + np.var(sobely))
