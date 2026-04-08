"""
Tests for Scoring Module
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.scoring import (
    calculate_final_score,
    get_grade_letter,
    get_verdict_level,
    calculate_percentage,
    is_passing,
    generate_feedback
)


def test_calculate_final_score():
    """Test final score calculation with default weights"""
    similarity = 0.8
    logic_flow = 0.7
    contradiction = 0.9
    
    score = calculate_final_score(similarity, logic_flow, contradiction)
    
    assert 0 <= score <= 1
    assert isinstance(score, float)
    # With weights 0.5, 0.3, 0.2: should be around 0.79
    assert 0.75 <= score <= 0.85


def test_calculate_final_score_custom_weights():
    """Test with custom weights"""
    weights = {
        'similarity': 0.6,
        'logic_flow': 0.3,
        'contradiction': 0.1
    }
    
    score = calculate_final_score(0.8, 0.7, 0.9, weights)
    
    assert 0 <= score <= 1
    # Should be around 0.77


def test_calculate_final_score_edge_cases():
    """Test with edge case values"""
    # All perfect scores
    score = calculate_final_score(1.0, 1.0, 1.0)
    assert score == 1.0
    
    # All zero scores
    score = calculate_final_score(0.0, 0.0, 0.0)
    assert score == 0.0
    
    # Values out of range (should be clamped)
    score = calculate_final_score(1.5, -0.5, 0.8)
    assert 0 <= score <= 1


def test_get_grade_letter():
    """Test grade letter assignment"""
    assert get_grade_letter(0.97) == "A+"
    assert get_grade_letter(0.92) == "A"
    assert get_grade_letter(0.87) == "A-"
    assert get_grade_letter(0.82) == "B+"
    assert get_grade_letter(0.77) == "B"
    assert get_grade_letter(0.67) == "C+"
    assert get_grade_letter(0.55) == "C-"
    assert get_grade_letter(0.45) == "F"


def test_get_verdict_level():
    """Test verdict level categorization"""
    assert get_verdict_level(0.90) == "excellent"
    assert get_verdict_level(0.75) == "good"
    assert get_verdict_level(0.60) == "needs-improvement"
    assert get_verdict_level(0.40) == "poor"


def test_calculate_percentage():
    """Test percentage conversion"""
    assert calculate_percentage(0.85) == 85.0
    assert calculate_percentage(0.5) == 50.0
    assert calculate_percentage(1.0) == 100.0
    assert calculate_percentage(0.0) == 0.0


def test_is_passing():
    """Test passing threshold check"""
    # Assuming pass_score is 0.6 in config
    assert is_passing(0.7) == True
    assert is_passing(0.6) == True
    assert is_passing(0.5) == False


def test_generate_feedback_excellent():
    """Test feedback generation for excellent answer"""
    feedback = generate_feedback(
        final_score=0.90,
        similarity_score=0.85,
        logic_flow_score=0.88,
        contradiction_score=0.95,
        relevance_flag='relevant'
    )
    
    assert 'overall' in feedback
    assert 'strengths' in feedback
    assert 'improvements' in feedback
    assert feedback['grade'] in ['A+', 'A', 'A-']
    assert feedback['passed'] == True
    assert len(feedback['strengths']) > 0


def test_generate_feedback_poor():
    """Test feedback generation for poor answer"""
    feedback = generate_feedback(
        final_score=0.40,
        similarity_score=0.35,
        logic_flow_score=0.40,
        contradiction_score=0.50,
        relevance_flag='irrelevant'
    )
    
    assert feedback['grade'] in ['D', 'F']
    assert feedback['passed'] == False
    assert len(feedback['improvements']) > 0


def test_generate_feedback_mixed():
    """Test feedback with mixed scores"""
    feedback = generate_feedback(
        final_score=0.65,
        similarity_score=0.85,  # Strong
        logic_flow_score=0.50,  # Weak
        contradiction_score=0.60,  # Weak
        relevance_flag='relevant'
    )
    
    assert len(feedback['strengths']) > 0
    assert len(feedback['improvements']) > 0
    # Should suggest improvements for logic and contradictions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
