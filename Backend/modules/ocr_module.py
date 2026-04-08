"""
Step 3: OCR Module - Text Extraction from Images
Supports multiple OCR engines:
- Azure Computer Vision (Read API) - Cloud OCR
- Local TrOCR (Microsoft handwriting model) - Recommended for offline
- Tesseract (fallback)
"""

import os
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import io
from loguru import logger
import yaml
import time

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def extract_text_azure(image: np.ndarray) -> Tuple[str, float]:
    """
    Extract text from image using Azure Computer Vision Read API.
    
    Args:
        image: Preprocessed image (numpy array or PIL Image)
        
    Returns:
        Tuple of (extracted_text, confidence_score)
    """
    try:
        from modules.azure_ocr import azure_extract_text
        
        start_time = time.time()
        
        # Convert to PIL Image if numpy array
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        # Get Azure config
        ocr_config = config.get('ocr', {})
        endpoint = os.getenv('AZURE_OCR_ENDPOINT') or ocr_config.get('azure_endpoint')
        key = os.getenv('AZURE_OCR_KEY') or ocr_config.get('azure_key')
        timeout = ocr_config.get('azure_timeout', 30)
        
        # Perform OCR with Azure
        text, confidence = azure_extract_text(
            image_bytes,
            endpoint=endpoint,
            key=key,
            timeout=timeout
        )
        
        elapsed = time.time() - start_time
        logger.info(
            f"Azure OCR extracted {len(text)} characters "
            f"(confidence: {confidence:.3f}, time: {elapsed:.2f}s)"
        )
        
        return text, confidence
        
    except Exception as e:
        logger.error(f"Azure OCR extraction failed: {str(e)}")
        raise


def extract_text_trocr(image: np.ndarray) -> Tuple[str, float]:
    """
    Extract text from image using local TrOCR model.
    
    Args:
        image: Preprocessed image (numpy array or PIL Image)
        
    Returns:
        Tuple of (extracted_text, confidence_score)
    """
    try:
        from modules.models.trocr_loader import trocr_ocr_image
        
        start_time = time.time()
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Get model config
        model_config = config.get('models', {})
        device = model_config.get('device', 'cpu')
        model_path = model_config.get('trocr_local_path', None)
        
        # Perform OCR with TrOCR
        text, confidence = trocr_ocr_image(
            pil_image,
            model_path=model_path,
            device=device
        )
        
        elapsed = time.time() - start_time
        logger.info(
            f"TrOCR extracted {len(text)} characters "
            f"(confidence: {confidence:.3f}, time: {elapsed:.2f}s)"
        )
        
        return text, confidence
        
    except Exception as e:
        logger.error(f"TrOCR extraction failed: {str(e)}")
        raise


def extract_text_from_image(image: np.ndarray, method: str = "auto") -> str:
    """
    Extract text from preprocessed image using configured OCR method.
    Supports intelligent fallback chain: TrOCR → Azure → Tesseract
    
    Args:
        image: Preprocessed image (numpy array or PIL Image)
        method: OCR method to use:
            - "auto": Try local_trocr, then azure, then tesseract
            - "local_trocr": Use local TrOCR model
            - "azure": Use Azure Computer Vision OCR
            - "tesseract": Use Tesseract OCR
            - "trocr": Alias for local_trocr
        
    Returns:
        Extracted text as string
    """
    try:
        # Get OCR provider from config
        ocr_config = config.get('ocr', {})
        provider = ocr_config.get('provider', 'auto')
        
        # Override with method parameter if provided
        if method != "auto":
            provider = method
        
        # Normalize provider name
        if provider == "trocr":
            provider = "local_trocr"
        
        logger.debug(f"OCR method: {provider}")
        
        # Try local TrOCR first if configured or auto
        if provider in ["local_trocr", "auto"]:
            try:
                text, confidence = extract_text_trocr(image)
                
                # Check if confidence is acceptable
                min_confidence = ocr_config.get('min_confidence', 0.3)
                if confidence >= min_confidence or provider == "local_trocr":
                    return text.strip()
                else:
                    logger.warning(
                        f"TrOCR confidence {confidence:.3f} below threshold {min_confidence}, "
                        f"trying fallback..."
                    )
            except Exception as e:
                logger.warning(f"TrOCR failed: {str(e)}, trying fallback...")
                if provider == "local_trocr":
                    # If explicitly requested local_trocr, raise error
                    raise
        
        # Try Azure OCR if auto or explicitly requested
        if provider in ["azure", "auto"]:
            try:
                text, confidence = extract_text_azure(image)
                logger.info(f"Azure OCR extracted {len(text)} characters")
                return text.strip()
            except Exception as e:
                logger.warning(f"Azure OCR failed: {str(e)}, trying Tesseract...")
                if provider == "azure":
                    # If explicitly requested azure, raise error
                    raise
        
        # Fallback to Tesseract
        logger.info("Using Tesseract OCR as fallback")
        return extract_text_tesseract(image)
        
    except Exception as e:
        logger.error(f"All OCR methods failed: {str(e)}")
        # Last resort: try Tesseract
        try:
            return extract_text_tesseract(image)
        except:
            raise Exception(f"OCR extraction completely failed: {str(e)}")


def extract_text_tesseract(image: np.ndarray) -> str:
    """
    Fallback OCR using Tesseract (if Vision API fails)
    
    Args:
        image: Preprocessed image
        
    Returns:
        Extracted text
    """
    try:
        import pytesseract
        from PIL import Image
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Perform OCR
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(pil_image, config=custom_config, lang='eng')
        
        logger.info(f"Tesseract OCR extracted {len(text)} characters")
        
        return text.strip()
        
    except ImportError:
        logger.error("Tesseract not installed. Install with: pip install pytesseract")
        raise Exception("OCR failed: No OCR engine available")
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {str(e)}")
        raise


def extract_text_with_layout(image: np.ndarray) -> dict:
    """
    Extract text with simple layout information.
    Note: Advanced layout extraction is only available with Azure or TrOCR.
    
    Returns:
        Dict with text and basic structure
    """
    try:
        # Use primary OCR method
        text = extract_text_from_image(image)
        
        # Split into simple paragraphs (by double newlines)
        paragraphs = []
        for para in text.split('\n\n'):
            if para.strip():
                paragraphs.append({
                    "text": para.strip(),
                    "confidence": 0.9  # Placeholder
                })
        
        return {
            "text": text,
            "paragraphs": paragraphs
        }
        
    except Exception as e:
        logger.error(f"Layout extraction failed: {str(e)}")
        return {"text": "", "paragraphs": []}
