"""
Test SBERT Loader Module
Tests local SBERT model loading and embedding functionality
"""

import sys
from pathlib import Path

# Add Backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

import pytest
import numpy as np
from modules.models.sbert_loader import (
    get_sbert_model,
    embed_text,
    embed_text_cached,
    embed_sentences_split,
    compute_similarity,
    get_model_info,
    clear_cache,
    unload_model
)


class TestSBERTLoader:
    """Test suite for SBERT loader module"""
    
    def test_model_loading(self):
        """Test that SBERT model loads successfully"""
        model = get_sbert_model(device='cpu')
        assert model is not None, "Model should load successfully"
        
        # Test singleton - should return same instance
        model2 = get_sbert_model(device='cpu')
        assert model is model2, "Should return same model instance (singleton)"
    
    def test_model_info(self):
        """Test model information retrieval"""
        # Load model first
        get_sbert_model(device='cpu')
        
        info = get_model_info()
        
        assert info['model_loaded'] == True, "Model should be marked as loaded"
        assert info['device'] == 'cpu', "Device should be CPU"
        assert info['local'] == True, "Should use local model"
        assert info['embedding_dimension'] == 384, "Should be 384-dim for all-MiniLM-L6-v2"
    
    def test_embed_single_text(self):
        """Test embedding a single text"""
        text = "This is a test sentence."
        
        embedding = embed_text(text, device='cpu')
        
        assert isinstance(embedding, np.ndarray), "Should return numpy array"
        assert embedding.shape == (384,), f"Should be 384-dim, got {embedding.shape}"
        assert embedding.dtype == np.float32, "Should be float32 dtype"
        
        # Check normalized (unit length)
        norm = np.linalg.norm(embedding)
        assert 0.99 <= norm <= 1.01, f"Should be normalized, got norm={norm:.3f}"
        
        print(f"\nSingle embedding shape: {embedding.shape}, norm: {norm:.3f}")
    
    def test_embed_multiple_texts(self):
        """Test embedding multiple texts (batch)"""
        texts = [
            "Hello world",
            "Machine learning is fascinating",
            "Natural language processing"
        ]
        
        embeddings = embed_text(texts, device='cpu', batch_size=8)
        
        assert isinstance(embeddings, np.ndarray), "Should return numpy array"
        assert embeddings.shape == (3, 384), f"Should be (3, 384), got {embeddings.shape}"
        
        # Check each embedding is normalized
        for i, emb in enumerate(embeddings):
            norm = np.linalg.norm(emb)
            assert 0.99 <= norm <= 1.01, f"Embedding {i} should be normalized"
        
        print(f"\nBatch embeddings shape: {embeddings.shape}")
    
    def test_cached_embedding(self):
        """Test cached embedding functionality"""
        text = "Cache this sentence"
        
        # First call - should compute
        emb1 = embed_text_cached(text)
        
        # Second call - should use cache
        emb2 = embed_text_cached(text)
        
        # Should be identical
        assert np.allclose(emb1, emb2), "Cached embeddings should be identical"
        
        print(f"\nCached embedding works correctly")
    
    def test_similarity_computation(self):
        """Test cosine similarity computation"""
        text1 = "The cat sits on the mat"
        text2 = "A feline rests on the rug"
        text3 = "Quantum physics is complex"
        
        # Similar texts should have higher similarity
        sim_similar = compute_similarity(text1, text2)
        sim_different = compute_similarity(text1, text3)
        
        assert 0.0 <= sim_similar <= 1.0, "Similarity should be between 0 and 1"
        assert 0.0 <= sim_different <= 1.0, "Similarity should be between 0 and 1"
        assert sim_similar > sim_different, "Similar texts should have higher similarity"
        
        print(f"\nSimilarity (similar): {sim_similar:.3f}")
        print(f"Similarity (different): {sim_different:.3f}")
    
    def test_sentence_splitting(self):
        """Test sentence splitting and embedding"""
        long_text = (
            "This is the first sentence. "
            "Here is another sentence. "
            "And a third one for good measure."
        )
        
        sentence_embeddings = embed_sentences_split(long_text, device='cpu')
        
        assert isinstance(sentence_embeddings, list), "Should return list"
        assert len(sentence_embeddings) == 3, f"Should split into 3 sentences, got {len(sentence_embeddings)}"
        
        for i, emb in enumerate(sentence_embeddings):
            assert emb.shape == (384,), f"Sentence {i} should be 384-dim"
        
        print(f"\nSplit into {len(sentence_embeddings)} sentence embeddings")
    
    def test_empty_text_handling(self):
        """Test handling of empty text"""
        # Empty string
        empty_emb = embed_text("", device='cpu')
        assert empty_emb.shape == (384,), "Should handle empty text"
        
        # Whitespace only
        whitespace_emb = embed_text("   ", device='cpu')
        assert whitespace_emb.shape == (384,), "Should handle whitespace"
        
        print(f"\nEmpty text handling: OK")
    
    def test_cache_clearing(self):
        """Test cache clearing functionality"""
        text = "Test cache clearing"
        
        # Generate cached embedding
        embed_text_cached(text)
        
        # Clear cache
        clear_cache()
        
        # Should work after clearing
        emb = embed_text_cached(text)
        assert emb.shape == (384,), "Should work after cache clear"
        
        print(f"\nCache clearing: OK")
    
    def test_model_unload(self):
        """Test model unloading functionality"""
        # Load model
        get_sbert_model(device='cpu')
        
        # Check loaded
        info = get_model_info()
        assert info['model_loaded'] == True
        
        # Unload
        unload_model()
        
        # Check unloaded
        info = get_model_info()
        assert info['model_loaded'] == False


def test_semantic_search_demo():
    """
    Demonstration of semantic search using SBERT
    """
    print("\n" + "="*60)
    print("SBERT Semantic Search Demo")
    print("="*60)
    
    # Query and corpus
    query = "How does machine learning work?"
    
    corpus = [
        "Machine learning uses algorithms to learn from data",
        "The weather today is sunny and warm",
        "Artificial intelligence mimics human intelligence",
        "I like pizza and pasta",
        "Deep learning is a subset of machine learning"
    ]
    
    print(f"\nQuery: '{query}'")
    print(f"\nSearching in corpus of {len(corpus)} documents...")
    
    # Compute similarities
    similarities = []
    for i, doc in enumerate(corpus):
        sim = compute_similarity(query, doc)
        similarities.append((i, doc, sim))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 3 Results:")
    for rank, (idx, doc, sim) in enumerate(similarities[:3], 1):
        print(f"{rank}. (Score: {sim:.3f}) {doc}")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)


def test_embedding_statistics():
    """
    Analyze embedding statistics
    """
    print("\n" + "="*60)
    print("Embedding Statistics Analysis")
    print("="*60)
    
    test_texts = [
        "Short",
        "This is a medium length sentence",
        "This is a much longer sentence with many more words and details about various topics",
    ]
    
    for text in test_texts:
        emb = embed_text(text, device='cpu')
        
        print(f"\nText: '{text[:50]}...' ({len(text)} chars)")
        print(f"  Embedding shape: {emb.shape}")
        print(f"  L2 norm: {np.linalg.norm(emb):.3f}")
        print(f"  Mean: {np.mean(emb):.3f}")
        print(f"  Std: {np.std(emb):.3f}")
        print(f"  Min: {np.min(emb):.3f}")
        print(f"  Max: {np.max(emb):.3f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run tests with pytest or directly
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            test_semantic_search_demo()
        elif sys.argv[1] == "stats":
            test_embedding_statistics()
    else:
        # Run pytest tests
        print("Running SBERT tests with pytest...")
        print("Options:")
        print("  python test_sbert_loader.py demo   - Run semantic search demo")
        print("  python test_sbert_loader.py stats  - Show embedding statistics")
        pytest.main([__file__, "-v", "-s"])
