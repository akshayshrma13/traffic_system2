"""Image Preprocessing Module - Extracted from preprocessing.ipynb"""

import cv2
import numpy as np
from .config import Config


def undistort_frame(frame):
    """
    Removes barrel distortion while preserving original resolution.
    
    Args:
        frame: Input image (BGR)
    
    Returns:
        Undistorted image with original dimensions
    """
    h, w = frame.shape[:2]
    K = np.array([[w * 0.8, 0, w // 2], 
                  [0, w * 0.8, h // 2], 
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([-0.15, 0.05, 0, 0], dtype=np.float32) 
    
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        K, D, (w, h), 
        Config.PREPROCESSING_PARAMS["undistort"]["alpha"], 
        (w, h)
    )
    undistorted = cv2.undistort(frame, K, D, None, new_camera_matrix)
    
    return undistorted


def enhance_lighting(frame):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    to normalize lighting without amplifying noise.
    
    Args:
        frame: Input image (BGR)
    
    Returns:
        Enhanced image
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(
        clipLimit=Config.PREPROCESSING_PARAMS["enhance_lighting"]["clipLimit"],
        tileGridSize=Config.PREPROCESSING_PARAMS["enhance_lighting"]["tileGridSize"]
    )
    cl = clahe.apply(l)
    
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)


def reduce_motion_blur(frame):
    """
    Reduces motion blur using unsharp masking.
    
    Args:
        frame: Input image (BGR)
    
    Returns:
        Sharpened image
    """
    kernel_size = Config.PREPROCESSING_PARAMS["reduce_motion_blur"]["kernel"]
    sigma = Config.PREPROCESSING_PARAMS["reduce_motion_blur"]["sigma"]
    
    gaussian_blur = cv2.GaussianBlur(frame, kernel_size, sigma)
    return cv2.addWeighted(frame, 1.5, gaussian_blur, -0.5, 0)


def dehaze_frame(frame, omega=None, t0=None):
    """
    Removes atmospheric haze using dark channel prior with bilateral filtering.
    
    Args:
        frame: Input image (BGR)
        omega: Haze removal strength (default from config)
        t0: Minimum transmission threshold (default from config)
    
    Returns:
        Dehazed image
    """
    if omega is None:
        omega = Config.PREPROCESSING_PARAMS["dehaze"]["omega"]
    if t0 is None:
        t0 = Config.PREPROCESSING_PARAMS["dehaze"]["t0"]
    
    img_normalized = frame.astype(np.float32)
    dark_channel = np.min(img_normalized, axis=2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dark_channel = cv2.erode(dark_channel, kernel)
    
    num_pixels = dark_channel.size
    num_brightest = max(int(num_pixels * 0.001), 1)
    indices = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]
    atmospheric_light = np.max(frame.reshape(-1, 3)[indices], axis=0)
    
    normalized = img_normalized / (atmospheric_light + 1e-6)
    transmission = 1 - omega * cv2.erode(np.min(normalized, axis=2), kernel)
    
    transmission = cv2.bilateralFilter(transmission, d=5, sigmaColor=0.1, sigmaSpace=5)
    transmission = np.maximum(transmission, t0)
    
    out = np.zeros_like(img_normalized)
    for i in range(3):
        out[:, :, i] = (img_normalized[:, :, i] - atmospheric_light[i]) / transmission + atmospheric_light[i]
    
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_dynamic_roi(frame, polygon_vertices=None):
    """
    Applies dynamic region-of-interest masking without changing image dimensions.
    Default: masks top 20% of frame (focuses on road)
    
    Args:
        frame: Input image (BGR)
        polygon_vertices: List of (x, y) tuples defining ROI polygon
    
    Returns:
        Masked image with same dimensions
    """
    if polygon_vertices is None:
        h, w = frame.shape[:2]
        top_pct = Config.PREPROCESSING_PARAMS["roi_mask"]["top_percentage"]
        polygon_vertices = [
            (0, int(h * top_pct)), 
            (w, int(h * top_pct)), 
            (w, h), 
            (0, h)
        ]
    
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    pts = np.array(polygon_vertices, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    return cv2.bitwise_and(frame, frame, mask=mask)


def preprocess_frame(image, apply_dehaze=False):
    """
    Complete preprocessing pipeline orchestrator.
    
    Steps:
        1. Undistort (remove lens barrel distortion)
        2. Enhance lighting (CLAHE)
        3. Reduce motion blur (unsharp mask)
        4. (Optional) Dehaze (remove atmospheric haze)
        5. Apply ROI mask (focus on road region)
    
    Args:
        image: Input image (BGR)
        apply_dehaze: Whether to apply dehazing (useful for foggy/rainy scenes)
    
    Returns:
        Tuple of (preprocessed_image, original_shape)
    """
    original_shape = image.shape
    
    # Apply preprocessing chain in order
    processed = undistort_frame(image)
    processed = enhance_lighting(processed)
    processed = reduce_motion_blur(processed)
    
    if apply_dehaze:
        processed = dehaze_frame(processed)
    
    processed = apply_dynamic_roi(processed)
    
    return processed, original_shape
