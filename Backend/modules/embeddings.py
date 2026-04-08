"""
Step 5: SBERT Embeddings Module
Generate sentence embeddings and compute similarity
Supports both local and remote SBERT models with intelligent fallback
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from loguru import logger
import yaml
import time
from functools import lru_cache

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Global model instance (lazy loading)
_embedding_model = None
_use_local = False


def get_embedding_model(force_reload: bool = False):
    """
    Get or initialize the SBERT model (singleton pattern).
    Supports both local and remote models with fallback.
    
    Args:
        force_reload: Force reload model even if already loaded
        
    Returns:
        SentenceTransformer model instance
    """
    global _embedding_model, _use_local
    
    if _embedding_model is not None and not force_reload:
        return _embedding_model
    
    try:
        # Check if local model should be used
        embeddings_config = config.get('embeddings', {})
        use_local = embeddings_config.get('use_local', False)
        
        if use_local:
            # Try loading local model first
            try:
                from modules.models.sbert_loader import get_sbert_model
                
                model_config = config.get('models', {})
                device = model_config.get('device', 'cpu')
                local_path = model_config.get('sbert_local_path', None)
                
                logger.info(f"Loading local SBERT model from: {local_path or 'default path'}")
                _embedding_model = get_sbert_model(
                    model_path=local_path,
                    device=device
                )
                _use_local = True
                logger.success("Local SBERT model loaded successfully")
                
                return _embedding_model
                
            except Exception as e:
                logger.warning(f"Local SBERT loading failed: {str(e)}, falling back to remote...")
                # Continue to remote loading
        
        # Load remote model (fallback or default)
        model_name = embeddings_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        logger.info(f"Loading remote SBERT model: {model_name}")
        
        _embedding_model = SentenceTransformer(model_name)
        _use_local = False
        logger.info("Remote SBERT model loaded successfully")
        
        return _embedding_model
        
    except Exception as e:
        logger.error(f"Failed to load SBERT model: {str(e)}")
        raise


@lru_cache(maxsize=1000)
def _generate_embeddings_cached(text: str) -> tuple:
    """
    Internal cached embedding function.
    Returns tuple for hashability in cache.
    """
    model = get_embedding_model()
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    return tuple(embedding.tolist())


def generate_embeddings(text: str, use_cache: bool = True) -> np.ndarray:
    """
    Generate SBERT embeddings for text with optional caching.
    
    Args:
        text: Input text (can be sentence or paragraph)
        use_cache: Use LRU cache for repeated texts (default: True)
        
    Returns:
        Embedding vector (numpy array, shape: (384,))
    """
    if not text or len(text.strip()) == 0:
        logger.warning("Empty text provided for embedding")
        return np.zeros(384)  # Default embedding size for MiniLM
    
    try:
        start_time = time.time()
        
        # Use cached version if enabled
        if use_cache:
            embedding_tuple = _generate_embeddings_cached(text)
            embedding = np.array(embedding_tuple, dtype=np.float32)
        else:
            model = get_embedding_model()
            embeddings_config = config.get('embeddings', {})
            batch_size = embeddings_config.get('batch_size', 32)
            
            # Generate embedding
            embedding = model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=batch_size
            )
        
        elapsed = time.time() - start_time
        
        logger.debug(
            f"Generated embedding of shape {embedding.shape} "
            f"(time: {elapsed*1000:.1f}ms, cached: {use_cache})"
        )
        
        return embedding
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        raise


def clear_embedding_cache():
    """Clear the LRU cache for embeddings"""
    _generate_embeddings_cached.cache_clear()
    logger.info("Embedding cache cleared")


def generate_sentence_embeddings(sentences: list, show_progress: bool = False) -> np.ndarray:
    """
    Generate embeddings for multiple sentences (batch processing).
    Optimized for performance with batching.
    
    Args:
        sentences: List of sentences
        show_progress: Show progress bar for large batches
        
    Returns:
        Array of embeddings (n_sentences x embedding_dim)
    """
    if not sentences:
        return np.array([])
    
    try:
        start_time = time.time()
        
        model = get_embedding_model()
        embeddings_config = config.get('embeddings', {})
        batch_size = embeddings_config.get('batch_size', 32)
        
        embeddings = model.encode(
            sentences,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            batch_size=batch_size
        )
        
        elapsed = time.time() - start_time
        
        logger.debug(
            f"Generated {len(embeddings)} sentence embeddings "
            f"(time: {elapsed:.2f}s, {len(embeddings)/elapsed:.1f} sent/s)"
        )
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {str(e)}")
        raise


def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Similarity score (0 to 1)
    """
    try:
        # Reshape if needed
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        # Ensure result is between 0 and 1
        similarity = float(max(0.0, min(1.0, similarity)))
        
        logger.debug(f"Computed similarity: {similarity:.4f}")
        
        return similarity
        
    except Exception as e:
        logger.error(f"Similarity computation failed: {str(e)}")
        raise


def compute_pairwise_similarities(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise similarities between multiple embeddings
    
    Args:
        embeddings: Array of embeddings (n x embedding_dim)
        
    Returns:
        Similarity matrix (n x n)
    """
    try:
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix
    except Exception as e:
        logger.error(f"Pairwise similarity computation failed: {str(e)}")
        raise


def get_most_similar_sentences(
    query_text: str,
    candidate_sentences: list,
    top_k: int = 5
) -> list:
    """
    Find most similar sentences to query
    
    Args:
        query_text: Query text
        candidate_sentences: List of candidate sentences
        top_k: Number of top results to return
        
    Returns:
        List of (sentence, similarity_score) tuples
    """
    try:
        # Generate embeddings
        query_embedding = generate_embeddings(query_text)
        candidate_embeddings = generate_sentence_embeddings(candidate_sentences)
        
        # Compute similarities
        similarities = []
        for i, candidate_emb in enumerate(candidate_embeddings):
            sim = compute_similarity(query_embedding, candidate_emb)
            similarities.append((candidate_sentences[i], sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
        
    except Exception as e:
        logger.error(f"Similar sentences search failed: {str(e)}")
        return []


def semantic_search(query: str, corpus: list, top_k: int = 5) -> list:
    """
    Semantic search in a corpus of documents
    
    Args:
        query: Search query
        corpus: List of documents/sentences
        top_k: Number of results to return
        
    Returns:
        List of (document, score) tuples
    """
    return get_most_similar_sentences(query, corpus, top_k)
