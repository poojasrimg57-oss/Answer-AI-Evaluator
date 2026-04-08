"""
Test Azure OCR Module
Tests Azure Computer Vision Read API integration
"""

import sys
from pathlib import Path

# Add Backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
import os
from PIL import Image, ImageDraw, ImageFont
import io
from modules.azure_ocr import azure_extract_text, test_azure_ocr, _extract_text_from_result, _calculate_confidence


def create_test_image(text="Azure OCR Test", size=(600, 100)):
    """
    Create a simple test image with text.
    """
    image = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    draw.text((50, 30), text, fill='black', font=font)
    
    return image


class TestAzureOCR:
    """Test suite for Azure OCR module"""
    
    def test_azure_credentials_available(self):
        """Test that Azure credentials are configured"""
        endpoint = os.getenv('AZURE_OCR_ENDPOINT')
        key = os.getenv('AZURE_OCR_KEY')
        
        assert endpoint is not None, "AZURE_OCR_ENDPOINT not set in environment"
        assert key is not None, "AZURE_OCR_KEY not set in environment"
        assert endpoint.startswith('https://'), "Endpoint should be HTTPS URL"
        
        print(f"\nAzure Endpoint: {endpoint}")
        print(f"Azure Key: {'*' * 20}...{key[-4:]}")
    
    def test_create_test_image(self):
        """Test image creation utility"""
        img = create_test_image("Hello World")
        
        assert img is not None
        assert img.size == (600, 100)
        assert img.mode == 'RGB'
    
    @pytest.mark.skipif(
        not os.getenv('AZURE_OCR_ENDPOINT') or not os.getenv('AZURE_OCR_KEY'),
        reason="Azure credentials not configured"
    )
    def test_azure_ocr_simple_text(self):
        """Test Azure OCR with simple generated text"""
        # Create test image
        test_image = create_test_image("Test Azure OCR")
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        # Perform OCR
        text, confidence = azure_extract_text(image_bytes)
        
        assert isinstance(text, str), "Should return string text"
        assert isinstance(confidence, float), "Should return float confidence"
        assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
        assert len(text) > 0, "Should extract some text"
        
        print(f"\nAzure OCR Result: '{text}'")
        print(f"Confidence: {confidence:.3f}")
    
    @pytest.mark.skipif(
        not os.getenv('AZURE_OCR_ENDPOINT') or not os.getenv('AZURE_OCR_KEY'),
        reason="Azure credentials not configured"
    )
    def test_azure_ocr_longer_text(self):
        """Test Azure OCR with longer text"""
        long_text = "The quick brown fox jumps over the lazy dog"
        test_image = create_test_image(long_text, size=(800, 100))
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
        
        # Perform OCR
        text, confidence = azure_extract_text(image_bytes)
        
        assert len(text) > 10, "Should extract substantial text"
        print(f"\nExtracted: '{text}'")
        print(f"Original:  '{long_text}'")
        print(f"Confidence: {confidence:.3f}")
    
    def test_azure_ocr_missing_credentials(self):
        """Test error handling when credentials missing"""
        # Temporarily clear credentials
        old_endpoint = os.getenv('AZURE_OCR_ENDPOINT')
        old_key = os.getenv('AZURE_OCR_KEY')
        
        os.environ.pop('AZURE_OCR_ENDPOINT', None)
        os.environ.pop('AZURE_OCR_KEY', None)
        
        try:
            test_image = create_test_image("Test")
            img_byte_arr = io.BytesIO()
            test_image.save(img_byte_arr, format='PNG')
            image_bytes = img_byte_arr.getvalue()
            
            with pytest.raises(ValueError, match="Azure OCR credentials not configured"):
                azure_extract_text(image_bytes)
                
        finally:
            # Restore credentials
            if old_endpoint:
                os.environ['AZURE_OCR_ENDPOINT'] = old_endpoint
            if old_key:
                os.environ['AZURE_OCR_KEY'] = old_key
    
    @pytest.mark.skipif(
        not os.getenv('AZURE_OCR_ENDPOINT') or not os.getenv('AZURE_OCR_KEY'),
        reason="Azure credentials not configured"
    )
    def test_azure_ocr_test_function(self):
        """Test the built-in test function"""
        result = test_azure_ocr()
        
        assert 'status' in result
        
        if result['status'] == 'success':
            assert 'text' in result
            assert 'confidence' in result
            print(f"\nTest function result: {result}")
        else:
            print(f"\nTest function error: {result.get('error')}")


def test_integration_demo():
    """
    Integration demo for Azure OCR
    """
    print("\n" + "="*60)
    print("Azure Computer Vision OCR Integration Demo")
    print("="*60)
    
    # Check credentials
    endpoint = os.getenv('AZURE_OCR_ENDPOINT')
    key = os.getenv('AZURE_OCR_KEY')
    region = os.getenv('AZURE_OCR_REGION')
    
    if not endpoint or not key:
        print("\n❌ Azure credentials not configured!")
        print("Set AZURE_OCR_ENDPOINT and AZURE_OCR_KEY in .env file")
        return
    
    print(f"\n✓ Credentials configured:")
    print(f"  Endpoint: {endpoint}")
    print(f"  Region: {region or 'not specified'}")
    print(f"  Key: {'*' * 40}...{key[-4:]}")
    
    # Create test image
    print("\n1. Creating test image...")
    test_text = "AnswerAI Evaluator - Azure OCR Integration"
    test_image = create_test_image(test_text, size=(900, 100))
    
    # Save image for inspection
    test_image.save("test_azure_ocr_output.png")
    print("   Saved to: test_azure_ocr_output.png")
    
    # Convert to bytes
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    image_bytes = img_byte_arr.getvalue()
    print(f"   Image size: {len(image_bytes)} bytes")
    
    # Perform OCR
    print("\n2. Sending to Azure Computer Vision...")
    try:
        text, confidence = azure_extract_text(image_bytes)
        
        print(f"\n3. Results:")
        print(f"   Status: SUCCESS ✓")
        print(f"   Extracted Text: '{text}'")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Characters: {len(text)}")
        print(f"   Words: {len(text.split())}")
        
    except Exception as e:
        print(f"\n3. Results:")
        print(f"   Status: FAILED ✗")
        print(f"   Error: {str(e)}")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        # Run integration demo
        test_integration_demo()
    else:
        # Run pytest tests
        print("Running Azure OCR tests with pytest...")
        print("\nNote: Tests require AZURE_OCR_ENDPOINT and AZURE_OCR_KEY in .env")
        print("To run integration demo: python test_azure_ocr.py demo")
        pytest.main([__file__, "-v", "-s"])
