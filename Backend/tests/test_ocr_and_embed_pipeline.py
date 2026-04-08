"""
Integration Test for OCR and Embedding Pipeline
Tests the complete workflow: Image -> OCR -> Preprocessing -> Embeddings -> Similarity
"""

import sys
from pathlib import Path

# Add Backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

# Import pipeline modules
from modules.models.trocr_loader import trocr_ocr_image
from modules.models.sbert_loader import embed_text, compute_similarity
from modules.text_preprocessing import clean_text, preprocess_text
from modules.ocr_module import extract_text_from_image


def create_handwritten_style_image(text, size=(800, 150)):
    """
    Create a test image with text that simulates handwriting style.
    """
    image = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a font
    try:
        # Try different font paths for different systems
        for font_path in ["arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:\\Windows\\Fonts\\arial.ttf"]:
            try:
                font = ImageFont.truetype(font_path, 42)
                break
            except:
                continue
        else:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Draw text
    draw.text((50, 50), text, fill='black', font=font)
    
    return image


class TestOCRAndEmbeddingPipeline:
    """Integration tests for OCR + Embedding pipeline"""
    
    def test_complete_pipeline_local_models(self):
        """
        Test complete pipeline using local TrOCR and SBERT models
        """
        print("\n" + "="*60)
        print("Testing Complete Local Model Pipeline")
        print("="*60)
        
        # 1. Create reference answer and student answer images
        reference_text = "Photosynthesis is the process by which plants convert light energy into chemical energy"
        student_text = "Photosynthesis converts sunlight into energy in plants"
        
        print(f"\n1. Reference Answer: {reference_text}")
        print(f"   Student Answer: {student_text}")
        
        # Create images
        ref_image = create_handwritten_style_image(reference_text, size=(900, 100))
        student_image = create_handwritten_style_image(student_text, size=(900, 100))
        
        # 2. Perform OCR
        print("\n2. Performing OCR with local TrOCR...")
        start_time = time.time()
        
        ref_ocr_text, ref_confidence = trocr_ocr_image(ref_image, device='cpu')
        student_ocr_text, student_confidence = trocr_ocr_image(student_image, device='cpu')
        
        ocr_time = time.time() - start_time
        print(f"   OCR Time: {ocr_time:.2f}s")
        print(f"   Reference OCR: '{ref_ocr_text}' (conf: {ref_confidence:.3f})")
        print(f"   Student OCR: '{student_ocr_text}' (conf: {student_confidence:.3f})")
        
        # 3. Preprocess text
        print("\n3. Preprocessing text...")
        ref_cleaned = clean_text(ref_ocr_text)
        student_cleaned = clean_text(student_ocr_text)
        
        ref_processed = preprocess_text(ref_cleaned)
        student_processed = preprocess_text(student_cleaned)
        
        print(f"   Reference processed: '{ref_processed}'")
        print(f"   Student processed: '{student_processed}'")
        
        # 4. Generate embeddings
        print("\n4. Generating SBERT embeddings...")
        start_time = time.time()
        
        ref_embedding = embed_text(ref_processed, device='cpu')
        student_embedding = embed_text(student_processed, device='cpu')
        
        embed_time = time.time() - start_time
        print(f"   Embedding Time: {embed_time:.2f}s")
        print(f"   Reference embedding shape: {ref_embedding.shape}")
        print(f"   Student embedding shape: {student_embedding.shape}")
        
        # 5. Compute similarity
        print("\n5. Computing semantic similarity...")
        similarity = compute_similarity(
            " ".join(ref_processed) if isinstance(ref_processed, list) else ref_processed,
            " ".join(student_processed) if isinstance(student_processed, list) else student_processed
        )
        
        print(f"   Semantic Similarity: {similarity:.3f}")
        
        # Assertions
        assert ref_embedding.shape == (384,), "Reference embedding should be 384-dim"
        assert student_embedding.shape == (384,), "Student embedding should be 384-dim"
        assert 0.0 <= similarity <= 1.0, "Similarity should be between 0 and 1"
        
        print("\n" + "="*60)
        print("✓ Pipeline test completed successfully!")
        print("="*60)
    
    def test_ocr_fallback_mechanism(self):
        """
        Test OCR fallback chain: TrOCR -> Vision API -> Tesseract
        """
        print("\n" + "="*60)
        print("Testing OCR Fallback Mechanism")
        print("="*60)
        
        test_image = create_handwritten_style_image("Test OCR fallback", size=(600, 100))
        
        # Test different methods
        methods = ["local_trocr", "auto"]
        
        for method in methods:
            print(f"\nTesting method: {method}")
            try:
                # Use the updated extract_text_from_image with method parameter
                result = extract_text_from_image(np.array(test_image), method=method)
                print(f"  Result: '{result}'")
                print(f"  Status: SUCCESS")
            except Exception as e:
                print(f"  Status: FAILED - {str(e)}")
        
        print("\n" + "="*60)
    
    def test_embedding_cache_performance(self):
        """
        Test embedding cache performance improvement
        """
        print("\n" + "="*60)
        print("Testing Embedding Cache Performance")
        print("="*60)
        
        test_text = "This is a test sentence for cache performance testing"
        
        # First call (no cache)
        start_time = time.time()
        emb1 = embed_text(test_text, device='cpu', use_cache=False)
        uncached_time = time.time() - start_time
        
        # Second call (with cache)
        start_time = time.time()
        emb2 = embed_text(test_text, device='cpu', use_cache=True)
        cached_time_1 = time.time() - start_time
        
        # Third call (should use cache)
        start_time = time.time()
        emb3 = embed_text(test_text, device='cpu', use_cache=True)
        cached_time_2 = time.time() - start_time
        
        print(f"\nFirst call (uncached): {uncached_time*1000:.2f}ms")
        print(f"Second call (cached):  {cached_time_1*1000:.2f}ms")
        print(f"Third call (cached):   {cached_time_2*1000:.2f}ms")
        print(f"Speedup: {uncached_time/cached_time_2:.1f}x")
        
        # Verify embeddings are identical
        assert np.allclose(emb1, emb2), "Embeddings should be identical"
        assert np.allclose(emb2, emb3), "Cached embeddings should be identical"
        
        print("\n✓ Cache working correctly!")
        print("="*60)
    
    def test_multiple_student_answers(self):
        """
        Test grading multiple student answers against one reference
        """
        print("\n" + "="*60)
        print("Testing Multiple Student Answers")
        print("="*60)
        
        reference = "Water boils at 100 degrees Celsius at sea level"
        
        students = [
            ("Student A", "Water boils at 100°C at standard atmospheric pressure"),
            ("Student B", "H2O reaches boiling point at 100 Celsius"),
            ("Student C", "The sky is blue because of light scattering"),
            ("Student D", "Water's boiling temperature is one hundred degrees")
        ]
        
        print(f"\nReference Answer:\n  {reference}\n")
        print("Student Answers and Similarity Scores:")
        
        ref_embedding = embed_text(reference, device='cpu')
        
        results = []
        for name, answer in students:
            student_embedding = embed_text(answer, device='cpu')
            similarity = compute_similarity(reference, answer)
            results.append((name, answer, similarity))
            
            print(f"\n{name}: (Score: {similarity:.3f})")
            print(f"  Answer: {answer}")
        
        # Sort by similarity
        results.sort(key=lambda x: x[2], reverse=True)
        
        print("\n\nRanking (Best to Worst):")
        for rank, (name, answer, sim) in enumerate(results, 1):
            grade = "A" if sim > 0.7 else "B" if sim > 0.5 else "C" if sim > 0.3 else "F"
            print(f"{rank}. {name}: {sim:.3f} (Grade: {grade})")
        
        print("\n" + "="*60)


def run_performance_benchmark():
    """
    Benchmark pipeline performance
    """
    print("\n" + "="*60)
    print("Pipeline Performance Benchmark")
    print("="*60)
    
    # Create test image
    test_text = "The quick brown fox jumps over the lazy dog"
    test_image = create_handwritten_style_image(test_text)
    
    n_iterations = 5
    
    # Benchmark OCR
    print(f"\n1. TrOCR Performance ({n_iterations} iterations)")
    ocr_times = []
    for i in range(n_iterations):
        start = time.time()
        text, conf = trocr_ocr_image(test_image, device='cpu')
        ocr_times.append(time.time() - start)
    
    print(f"   Average: {np.mean(ocr_times)*1000:.1f}ms")
    print(f"   Min: {np.min(ocr_times)*1000:.1f}ms")
    print(f"   Max: {np.max(ocr_times)*1000:.1f}ms")
    
    # Benchmark Embeddings
    print(f"\n2. SBERT Embedding Performance ({n_iterations} iterations)")
    embed_times = []
    for i in range(n_iterations):
        start = time.time()
        emb = embed_text(test_text, device='cpu', use_cache=False)
        embed_times.append(time.time() - start)
    
    print(f"   Average: {np.mean(embed_times)*1000:.1f}ms")
    print(f"   Min: {np.min(embed_times)*1000:.1f}ms")
    print(f"   Max: {np.max(embed_times)*1000:.1f}ms")
    
    # Total pipeline time
    total_time = np.mean(ocr_times) + np.mean(embed_times)
    print(f"\n3. Total Pipeline Time (OCR + Embedding)")
    print(f"   Average: {total_time*1000:.1f}ms")
    print(f"   Throughput: {1/total_time:.1f} evaluations/second")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "benchmark":
            run_performance_benchmark()
        elif sys.argv[1] == "full":
            # Run all tests manually
            test = TestOCRAndEmbeddingPipeline()
            test.test_complete_pipeline_local_models()
            test.test_ocr_fallback_mechanism()
            test.test_embedding_cache_performance()
            test.test_multiple_student_answers()
    else:
        # Run pytest tests
        print("Running OCR & Embedding Pipeline integration tests...")
        print("\nOptions:")
        print("  python test_ocr_and_embed_pipeline.py benchmark - Run performance benchmark")
        print("  python test_ocr_and_embed_pipeline.py full      - Run all tests")
        pytest.main([__file__, "-v", "-s"])
