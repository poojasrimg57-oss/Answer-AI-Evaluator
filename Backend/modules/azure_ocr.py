"""
Azure Computer Vision OCR Module
Uses Azure Cognitive Services Read API for text extraction from images
"""

import os
import time
import requests
from typing import Tuple
from loguru import logger
from PIL import Image
import io


def azure_extract_text(image_bytes: bytes, endpoint: str = None, key: str = None, timeout: int = 30) -> Tuple[str, float]:
    """
    Extract text from image using Azure Computer Vision Read API.
    
    Args:
        image_bytes: Image data as bytes
        endpoint: Azure OCR endpoint URL (from env if None)
        key: Azure OCR subscription key (from env if None)
        timeout: Maximum time to wait for result (seconds)
        
    Returns:
        Tuple of (extracted_text, confidence_score)
        
    Raises:
        Exception: If OCR fails or times out
    """
    try:
        # Get credentials from environment if not provided
        if endpoint is None:
            endpoint = os.getenv('AZURE_OCR_ENDPOINT')
        if key is None:
            key = os.getenv('AZURE_OCR_KEY')
        
        if not endpoint or not key:
            raise ValueError(
                "Azure OCR credentials not configured. "
                "Set AZURE_OCR_ENDPOINT and AZURE_OCR_KEY in .env file"
            )
        
        # Remove trailing slash from endpoint
        endpoint = endpoint.rstrip('/')
        
        # Construct the Read API URL (v3.2)
        read_url = f"{endpoint}/vision/v3.2/read/analyze"
        
        # Set headers
        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Content-Type': 'application/octet-stream'
        }
        
        # Parameters
        params = {
            'language': 'en',
            'readingOrder': 'natural'
        }
        
        logger.info("Sending image to Azure Computer Vision Read API...")
        start_time = time.time()
        
        # Submit the image for processing
        response = requests.post(
            read_url,
            headers=headers,
            params=params,
            data=image_bytes,
            timeout=10
        )
        
        if response.status_code != 202:
            error_msg = f"Azure OCR submission failed: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # Get the operation location from response headers
        operation_location = response.headers.get('Operation-Location')
        if not operation_location:
            raise Exception("Azure OCR did not return Operation-Location header")
        
        logger.info(f"Azure OCR job submitted. Polling for results...")
        
        # Poll for results
        poll_interval = 1  # seconds
        elapsed = 0
        
        while elapsed < timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval
            
            # Get the result
            result_response = requests.get(
                operation_location,
                headers={'Ocp-Apim-Subscription-Key': key},
                timeout=10
            )
            
            if result_response.status_code != 200:
                logger.warning(f"Azure OCR polling failed: {result_response.status_code}")
                continue
            
            result_json = result_response.json()
            status = result_json.get('status')
            
            if status == 'succeeded':
                # Extract text from results
                extracted_text = _extract_text_from_result(result_json)
                confidence = _calculate_confidence(result_json)
                
                processing_time = time.time() - start_time
                logger.info(
                    f"Azure OCR completed: {len(extracted_text)} chars "
                    f"(confidence: {confidence:.3f}, time: {processing_time:.2f}s)"
                )
                
                return extracted_text, confidence
                
            elif status == 'failed':
                error_msg = f"Azure OCR processing failed: {result_json.get('message', 'Unknown error')}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Status is still 'running' or 'notStarted', continue polling
            logger.debug(f"Azure OCR status: {status}, elapsed: {elapsed}s")
        
        # Timeout reached
        raise TimeoutError(f"Azure OCR timed out after {timeout} seconds")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Azure OCR network error: {str(e)}")
        raise Exception(f"Azure OCR network error: {str(e)}")
    except Exception as e:
        logger.error(f"Azure OCR failed: {str(e)}")
        raise


def _extract_text_from_result(result_json: dict) -> str:
    """
    Extract all text lines from Azure OCR result JSON.
    
    Args:
        result_json: Azure Read API result JSON
        
    Returns:
        Concatenated text string
    """
    try:
        text_lines = []
        
        # Navigate through the result structure
        analyze_result = result_json.get('analyzeResult', {})
        read_results = analyze_result.get('readResults', [])
        
        for page in read_results:
            lines = page.get('lines', [])
            for line in lines:
                text = line.get('text', '')
                if text:
                    text_lines.append(text)
        
        # Join all lines with spaces
        full_text = ' '.join(text_lines)
        
        logger.debug(f"Extracted {len(text_lines)} lines from Azure OCR result")
        
        return full_text.strip()
        
    except Exception as e:
        logger.error(f"Failed to extract text from Azure result: {str(e)}")
        return ""


def _calculate_confidence(result_json: dict) -> float:
    """
    Calculate average confidence score from Azure OCR result.
    
    Args:
        result_json: Azure Read API result JSON
        
    Returns:
        Average confidence score (0.0 to 1.0)
    """
    try:
        confidences = []
        
        analyze_result = result_json.get('analyzeResult', {})
        read_results = analyze_result.get('readResults', [])
        
        for page in read_results:
            lines = page.get('lines', [])
            for line in lines:
                # Azure Read API doesn't always provide confidence
                # If available, it's typically per word
                words = line.get('words', [])
                for word in words:
                    conf = word.get('confidence')
                    if conf is not None:
                        confidences.append(float(conf))
        
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
        else:
            # If no confidence scores available, assume high confidence
            # Azure Read API is generally reliable
            avg_confidence = 0.95
        
        return avg_confidence
        
    except Exception as e:
        logger.warning(f"Failed to calculate confidence: {str(e)}")
        return 0.0


def test_azure_ocr(image_path: str = None) -> dict:
    """
    Test Azure OCR with a sample image.
    
    Args:
        image_path: Path to test image (optional, creates sample if None)
        
    Returns:
        Dict with OCR results
    """
    try:
        if image_path is None:
            # Create a simple test image
            from PIL import Image, ImageDraw, ImageFont
            
            img = Image.new('RGB', (600, 100), color='white')
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 36)
            except:
                font = ImageFont.load_default()
            
            draw.text((50, 30), "Test Azure OCR", fill='black', font=font)
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            image_bytes = img_bytes.getvalue()
        else:
            # Read image file
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
        
        # Perform OCR
        text, confidence = azure_extract_text(image_bytes)
        
        return {
            'status': 'success',
            'text': text,
            'confidence': confidence,
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


if __name__ == "__main__":
    # Test the module
    logger.info("Testing Azure Computer Vision OCR...")
    
    result = test_azure_ocr()
    
    print("\n" + "="*60)
    print("Azure OCR Test Results")
    print("="*60)
    
    if result['status'] == 'success':
        print(f"Status: SUCCESS")
        print(f"Text: {result['text']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Length: {result['text_length']} chars")
        print(f"Words: {result['word_count']}")
    else:
        print(f"Status: FAILED")
        print(f"Error: {result['error']}")
    
    print("="*60)
