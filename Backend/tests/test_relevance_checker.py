"""
Tests for Relevance Checker Module
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.relevance_checker import (
    check_relevance,
    check_keyword_overlap,
    detect_off_topic
)
from modules.embeddings import generate_embeddings


def test_check_relevance_relevant():
    """Test relevance check with relevant answer"""
    question = "What is machine learning?"
    reference = "Machine learning is a subset of AI that enables systems to learn from data."
    student = "Machine learning allows computers to learn patterns from data automatically."
    
    # Generate embeddings
    student_emb = generate_embeddings(student)
    question_emb = generate_embeddings(question)
    reference_emb = generate_embeddings(reference)
    
    result = check_relevance(student_emb, question_emb, reference_emb, threshold=0.4)
    
    assert result['relevance_flag'] in ['relevant', 'irrelevant']
    assert 0 <= result['similarity_to_reference'] <= 1
    assert 0 <= result['similarity_to_question'] <= 1
    assert 'is_relevant' in result


def test_check_relevance_irrelevant():
    """Test relevance check with irrelevant answer"""
    question = "What is machine learning?"
    reference = "Machine learning is a subset of AI."
    student = "The weather is nice today and I like pizza."
    
    student_emb = generate_embeddings(student)
    question_emb = generate_embeddings(question)
    reference_emb = generate_embeddings(reference)
    
    result = check_relevance(student_emb, question_emb, reference_emb, threshold=0.5)
    
    # Should have low similarity scores
    assert result['similarity_to_reference'] < 0.5
    assert result['relevance_flag'] == 'irrelevant'


def test_keyword_overlap():
    """Test keyword overlap calculation"""
    student = "Machine learning uses data to train models"
    reference = "Machine learning algorithms learn from data"
    
    overlap = check_keyword_overlap(student, reference)
    
    assert 0 <= overlap <= 1
    assert overlap > 0.3  # Should have some overlap


def test_keyword_overlap_no_match():
    """Test keyword overlap with no matching words"""
    student = "The quick brown fox"
    reference = "Artificial intelligence systems"
    
    overlap = check_keyword_overlap(student, reference)
    
    assert overlap < 0.2  # Very low overlap


def test_detect_off_topic():
    """Test off-topic detection"""
    question = "Explain photosynthesis"
    student = "Photosynthesis is the process plants use to make food using sunlight"
    
    student_emb = generate_embeddings(student)
    question_emb = generate_embeddings(question)
    
    result = detect_off_topic(student, question, student_emb, question_emb)
    
    assert 'is_off_topic' in result
    assert result['is_off_topic'] == False
    assert result['similarity_score'] > 0.3


def test_detect_off_topic_positive():
    """Test detecting actually off-topic answer"""
    question = "Explain photosynthesis"
    student = "I went to the store yesterday and bought groceries"
    
    student_emb = generate_embeddings(student)
    question_emb = generate_embeddings(question)
    
    result = detect_off_topic(student, question, student_emb, question_emb)
    
    # Should be detected as off-topic
    assert result['similarity_score'] < 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
