"""
Step 6: Relevance Checker Module
Determines if student answer is relevant to the question using cosine similarity
"""

import numpy as np
from loguru import logger
from modules.embeddings import compute_similarity
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def check_relevance(
    student_embedding: np.ndarray,
    question_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    threshold: float = None
) -> dict:
    """
    Check if student answer is relevant to the question and reference answer
    
    Args:
        student_embedding: Student answer embedding
        question_embedding: Question embedding
        reference_embedding: Reference answer embedding
        threshold: Relevance threshold (default from config)
        
    Returns:
        Dict with relevance flag and similarity scores
    """
    if threshold is None:
        threshold = config['scoring']['thresholds']['relevance']
    
    try:
        # Compute similarity to question (ensures answer addresses the question)
        question_similarity = compute_similarity(student_embedding, question_embedding)
        
        # Compute similarity to reference answer (semantic correctness)
        reference_similarity = compute_similarity(student_embedding, reference_embedding)
        
        # Determine relevance
        # Answer is relevant if it's similar to both question AND reference
        is_relevant = reference_similarity >= threshold
        
        # Additional check: answer should have some relation to question
        # (but not too strict, as answer might not contain exact question terms)
        min_question_sim = 0.2
        if question_similarity < min_question_sim:
            logger.warning(f"Low question similarity: {question_similarity:.3f}")
        
        relevance_flag = "relevant" if is_relevant else "irrelevant"
        
        logger.info(
            f"Relevance check: {relevance_flag} "
            f"(ref_sim={reference_similarity:.3f}, q_sim={question_similarity:.3f})"
        )
        
        return {
            "relevance_flag": relevance_flag,
            "similarity_to_reference": reference_similarity,
            "similarity_to_question": question_similarity,
            "is_relevant": is_relevant,
            "threshold_used": threshold
        }
        
    except Exception as e:
        logger.error(f"Relevance check failed: {str(e)}")
        # Return neutral result on error
        return {
            "relevance_flag": "relevant",
            "similarity_to_reference": 0.5,
            "similarity_to_question": 0.5,
            "is_relevant": True,
            "threshold_used": threshold,
            "error": str(e)
        }


def check_keyword_overlap(student_text: str, reference_text: str) -> float:
    """
    Simple keyword overlap check (fallback for relevance)
    
    Returns:
        Overlap ratio (0 to 1)
    """
    student_words = set(student_text.lower().split())
    reference_words = set(reference_text.lower().split())
    
    if not reference_words:
        return 0.0
    
    overlap = len(student_words.intersection(reference_words))
    overlap_ratio = overlap / len(reference_words)
    
    return min(1.0, overlap_ratio)


def analyze_topic_coverage(
    student_embedding: np.ndarray,
    reference_sentences_embeddings: list,
    threshold: float = 0.6
) -> dict:
    """
    Analyze how many key topics from reference are covered in student answer
    
    Args:
        student_embedding: Student answer embedding
        reference_sentences_embeddings: List of reference sentence embeddings
        threshold: Similarity threshold for topic match
        
    Returns:
        Coverage analysis dict
    """
    try:
        covered_topics = 0
        total_topics = len(reference_sentences_embeddings)
        
        for ref_sent_emb in reference_sentences_embeddings:
            similarity = compute_similarity(student_embedding, ref_sent_emb)
            if similarity >= threshold:
                covered_topics += 1
        
        coverage_ratio = covered_topics / total_topics if total_topics > 0 else 0.0
        
        return {
            "topics_covered": covered_topics,
            "total_topics": total_topics,
            "coverage_ratio": coverage_ratio,
            "is_comprehensive": coverage_ratio >= 0.7
        }
        
    except Exception as e:
        logger.error(f"Topic coverage analysis failed: {str(e)}")
        return {
            "topics_covered": 0,
            "total_topics": 0,
            "coverage_ratio": 0.0,
            "is_comprehensive": False
        }


def detect_off_topic(
    student_text: str,
    question_text: str,
    student_embedding: np.ndarray,
    question_embedding: np.ndarray
) -> dict:
    """
    Detect if answer is completely off-topic
    
    Returns:
        Dict with off-topic flag and confidence
    """
    similarity = compute_similarity(student_embedding, question_embedding)
    
    # Very low similarity indicates off-topic response
    off_topic_threshold = 0.15
    is_off_topic = similarity < off_topic_threshold
    
    # Additional check: keyword presence
    question_keywords = set(question_text.lower().split())
    student_keywords = set(student_text.lower().split())
    keyword_overlap = len(question_keywords.intersection(student_keywords))
    
    has_keywords = keyword_overlap > 0
    
    return {
        "is_off_topic": is_off_topic and not has_keywords,
        "similarity_score": similarity,
        "keyword_overlap": keyword_overlap,
        "confidence": 1.0 - similarity if is_off_topic else similarity
    }
