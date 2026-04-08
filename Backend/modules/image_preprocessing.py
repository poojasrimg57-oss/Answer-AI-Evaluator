"""
Step 2: Image Preprocessing Module
Prepares images for OCR by applying grayscale, blur, and thresholding
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from loguru import logger
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess image for OCR
    
    Steps:
    1. Load image
    2. Resize if too large
    3. Convert to grayscale
    4. Apply Gaussian blur (noise reduction)
    5. Apply adaptive thresholding (binarization)
    
    Args:
        image_path: Path to input image
        
    Returns:
        Preprocessed image as numpy array
    """
    try:
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
        
        if image is None:
            raise ValueError("Could not load image")
        
        logger.debug(f"Original image shape: {image.shape}")
        
        # Resize if too large
        max_size = config['image_processing']['max_image_size']
        height, width = image.shape[:2]
        
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized to: {image.shape}")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur for noise reduction
        kernel_size = config['image_processing']['gaussian_blur_kernel']
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        
        # Apply adaptive thresholding for better text extraction
        block_size = config['image_processing']['adaptive_threshold']['block_size']
        c = config['image_processing']['adaptive_threshold']['c_constant']
        
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c
        )
        
        # Optional: Remove noise with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        denoised = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        
        logger.info("Image preprocessing completed successfully")
        
        return denoised
        
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Deskew image to correct rotation
    """
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return rotated


def remove_borders(image: np.ndarray) -> np.ndarray:
    """
    Remove black borders from scanned documents
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = image[y:y+h, x:x+w]
        return cropped
    return image
