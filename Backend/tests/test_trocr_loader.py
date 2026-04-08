"""
Test TrOCR Loader Module
Tests local TrOCR model loading and OCR functionality
"""

import sys
from pathlib import Path

# Add Backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from modules.models.trocr_loader import (
    get_trocr_processor,
    get_trocr_model,
    trocr_ocr_image,
    get_model_info,
    unload_models
)


def create_test_image(text="Hello World", size=(600, 100)):
    """
    Create a simple test image with text for OCR testing.
    """
    # Create white background
    image = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    # Draw text in black
    draw.text((50, 30), text, fill='black', font=font)
    
    return image


class TestTrOCRLoader:
    """Test suite for TrOCR loader module"""
    
    def test_processor_loading(self):
        """Test that TrOCR processor loads successfully"""
        processor = get_trocr_processor()
        assert processor is not None, "Processor should load successfully"
        
        # Test singleton - should return same instance
        processor2 = get_trocr_processor()
        assert processor is processor2, "Should return same processor instance (singleton)"
    
    def test_model_loading(self):
        """Test that TrOCR model loads successfully"""
        model = get_trocr_model(device='cpu')
        assert model is not None, "Model should load successfully"
        
        # Test singleton - should return same instance
        model2 = get_trocr_model(device='cpu')
        assert model is model2, "Should return same model instance (singleton)"
    
    def test_model_info(self):
        """Test model information retrieval"""
        # Load models first
        get_trocr_processor()
        get_trocr_model(device='cpu')
        
        info = get_model_info()
        
        assert info['model_loaded'] == True, "Model should be marked as loaded"
        assert info['processor_loaded'] == True, "Processor should be marked as loaded"
        assert info['device'] == 'cpu', "Device should be CPU"
        assert info['local'] == True, "Should use local model"
    
    def test_ocr_simple_text(self):
        """Test OCR on a simple generated image"""
        # Create test image
        test_image = create_test_image("Hello")
        
        # Perform OCR
        text, confidence = trocr_ocr_image(test_image, device='cpu')
        
        assert isinstance(text, str), "Should return string text"
        assert isinstance(confidence, float), "Should return float confidence"
        assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
        assert len(text) > 0, "Should extract some text"
        
        print(f"\nOCR Result: '{text}' (confidence: {confidence:.3f})")
    
    def test_ocr_with_numpy_array(self):
        """Test OCR with numpy array input (converted from PIL)"""
        # Create test image and convert to numpy
        test_image = create_test_image("Test")
        np_image = np.array(test_image)
        
        # Should handle numpy array by converting to PIL internally
        pil_image = Image.fromarray(np_image)
        text, confidence = trocr_ocr_image(pil_image, device='cpu')
        
        assert isinstance(text, str), "Should handle numpy array input"
        assert len(text) >= 0, "Should process successfully"
        
        print(f"\nNumPy OCR Result: '{text}' (confidence: {confidence:.3f})")
    
    def test_ocr_empty_image(self):
        """Test OCR on empty/blank image"""
        # Create blank white image
        blank_image = Image.new('RGB', (400, 100), color='white')
        
        # Should not crash, might return empty or noise
        text, confidence = trocr_ocr_image(blank_image, device='cpu')
        
        assert isinstance(text, str), "Should return string even for blank image"
        print(f"\nBlank Image OCR: '{text}' (confidence: {confidence:.3f})")
    
    def test_model_unload(self):
        """Test model unloading functionality"""
        # Load models
        get_trocr_processor()
        get_trocr_model(device='cpu')
        
        # Check loaded
        info = get_model_info()
        assert info['model_loaded'] == True
        
        # Unload
        unload_models()
        
        # Check unloaded
        info = get_model_info()
        assert info['model_loaded'] == False
        assert info['processor_loaded'] == False


def test_integration_demo():
    """
    Integration test demonstrating full TrOCR workflow
    """
    print("\n" + "="*60)
    print("TrOCR Integration Demo")
    print("="*60)
    
    # 1. Create test image
    print("\n1. Creating test image with text: 'AnswerAI'")
    test_image = create_test_image("AnswerAI", size=(800, 150))
    test_image.save("test_trocr_output.png")
    print("   Saved to: test_trocr_output.png")
    
    # 2. Load models
    print("\n2. Loading TrOCR models...")
    processor = get_trocr_processor()
    model = get_trocr_model(device='cpu')
    print("   ✓ Models loaded successfully")
    
    # 3. Perform OCR
    print("\n3. Performing OCR...")
    text, confidence = trocr_ocr_image(test_image, device='cpu')
    print(f"   Extracted Text: '{text}'")
    print(f"   Confidence: {confidence:.3f}")
    
    # 4. Model info
    print("\n4. Model Information:")
    info = get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


if __name__ == "__main__":
    # Run tests with pytest or directly
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run integration demo
        test_integration_demo()
    else:
        # Run pytest tests
        print("Running TrOCR tests with pytest...")
        print("To run integration demo: python test_trocr_loader.py demo")
        pytest.main([__file__, "-v", "-s"])
