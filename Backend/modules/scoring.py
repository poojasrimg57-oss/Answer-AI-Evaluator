"""
Step 9: Final Scoring Module
Calculates weighted final score from component scores
Now supports ML model inference and enhanced AI evaluation
"""

from loguru import logger
import yaml
import numpy as np
from pathlib import Path

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ML Model components (loaded on demand)
_ml_model = None
_ml_scaler = None
_ml_feature_names = None
_ml_model_available = False


def load_ml_model():
    """
    Load trained ML model if available
    
    Returns:
        bool: True if model loaded successfully
    """
    global _ml_model, _ml_scaler, _ml_feature_names, _ml_model_available
    
    if _ml_model_available:
        return True  # Already loaded
    
    try:
        # Check if model files exist
        model_path = Path("models/scoring_model.pkl")
        if not model_path.exists():
            logger.info("ML model not found, using weighted scoring as default")
            return False
        
        # Import here to avoid dependency issues
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from model_training.save_load_model import load_scoring_model
        
        # Load model artifacts
        artifacts = load_scoring_model()
        _ml_model = artifacts['model']
        _ml_scaler = artifacts['scaler']
        _ml_feature_names = artifacts['feature_names']
        _ml_model_available = True
        
        logger.info("✅ ML scoring model loaded successfully")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to load ML model: {str(e)}")
        logger.info("Falling back to weighted scoring")
        return False


def calculate_ml_score(features: dict) -> float:
    """
    Calculate score using trained ML model
    
    Args:
        features: Dictionary of extracted features
        
    Returns:
        Predicted score (normalized to 0-1)
    """
    global _ml_model, _ml_scaler, _ml_feature_names
    
    # Load model if not already loaded
    if not _ml_model_available:
        if not load_ml_model():
            return None  # Model not available
    
    try:
        # Separate embedding from other features
        embedding = features.pop('essay_embedding', None)
        
        # Construct feature vector in correct order
        feature_vector = []
        for name in _ml_feature_names:
            if name.startswith('emb_'):
                continue  # Handle embeddings separately
            feature_vector.append(features.get(name, 0.0))
        
        # Add embedding dimensions
        if embedding is not None:
            feature_vector.extend(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)
        
        # Convert to numpy array
        X = np.array([feature_vector])
        
        # Apply scaling
        if _ml_scaler is not None:
            X = _ml_scaler.transform(X)
        
        # Predict
        score = _ml_model.predict(X)[0]
        
        # Normalize to 0-1 range (ASAP scores typically 0-4 or 0-5)
        # Detect score range from prediction
        if score > 5:
            score = score / 10  # Assume 0-10 scale
        elif score > 1:
            score = score / 5  # Assume 0-5 scale
        
        # Clamp to valid range
        score = max(0.0, min(1.0, score))
        
        logger.info(f"ML model prediction: {score:.4f}")
        return float(score)
        
    except Exception as e:
        logger.error(f"ML score calculation failed: {str(e)}")
        return None


def calculate_final_score(
    similarity_score: float,
    logic_flow_score: float,
    contradiction_score: float,
    weights: dict = None,
    features: dict = None,
    use_ml_model: bool = True,
    question: str = None,
    reference_answer: str = None,
    student_answer: str = None,
    relevance_score: float = 1.0
) -> float:
    """
    Calculate weighted final score with enhanced AI evaluation
    
    Scoring priority:
    1. Enhanced AI Model (if available) - Most accurate
    2. ML Model (if trained and available)
    3. Weighted scoring (traditional fallback)
    
    Args:
        similarity_score: Semantic similarity score (0-1)
        logic_flow_score: Logic flow score (0-1)
        contradiction_score: Contradiction score (0-1, higher = fewer contradictions)
        weights: Custom weights (default from config)
        features: Full feature dictionary (for ML model)
        use_ml_model: Whether to use ML model if available
        question: Exam question text (for enhanced evaluation)
        reference_answer: Reference answer text (for enhanced evaluation)
        student_answer: Student answer text (for enhanced evaluation)
        
    Returns:
        Final score (0-1)
    """
    # Try enhanced AI model first if all text inputs provided
    if question and reference_answer and student_answer:
        try:
            from modules.enhanced_scorer import get_enhanced_scores
            
            enhanced_result = get_enhanced_scores(
                question=question,
                reference_answer=reference_answer,
                student_answer=student_answer,
                initial_similarity=similarity_score,
                initial_logic=logic_flow_score,
                initial_contradiction=contradiction_score,
                initial_relevance=relevance_score
            )
            
            if enhanced_result and isinstance(enhanced_result, dict):
                logger.info(f"✅ Using enhanced AI evaluation: {enhanced_result.get('final_score', 0):.4f}")
                # Return complete enhanced scores dictionary
                return {
                    'relevance_score': enhanced_result.get('relevance_score', relevance_score),
                    'similarity_score': enhanced_result.get('similarity_score', similarity_score),
                    'logic_flow_score': enhanced_result.get('logic_flow_score', logic_flow_score),
                    'contradiction_score': enhanced_result.get('contradiction_score', contradiction_score),
                    'final_score': enhanced_result.get('final_score', 0.0)
                }
                
        except Exception as e:
            logger.debug(f"Enhanced evaluation not available: {str(e)}")
    
    # Try ML model second if enabled and features provided
    if use_ml_model and features is not None:
        ml_score = calculate_ml_score(features.copy())
        if ml_score is not None:
            logger.info(f"Using ML model score: {ml_score:.4f}")
            return {
                'similarity_score': similarity_score,
                'logic_flow_score': logic_flow_score,
                'contradiction_score': contradiction_score,
                'final_score': ml_score
            }
    
    # Fallback to weighted scoring
    if weights is None:
        weights = config['scoring']['weights']
    
    try:
        # Validate inputs
        similarity_score = max(0.0, min(1.0, similarity_score))
        logic_flow_score = max(0.0, min(1.0, logic_flow_score))
        contradiction_score = max(0.0, min(1.0, contradiction_score))
        
        # Calculate weighted score
        w_similarity = weights.get('similarity', 0.5)
        w_logic = weights.get('logic_flow', 0.3)
        w_contradiction = weights.get('contradiction', 0.2)
        
        # Normalize weights (ensure they sum to 1)
        total_weight = w_similarity + w_logic + w_contradiction
        if total_weight != 1.0:
            logger.warning(f"Weights sum to {total_weight}, normalizing...")
            w_similarity /= total_weight
            w_logic /= total_weight
            w_contradiction /= total_weight
        
        final_score = (
            w_similarity * similarity_score +
            w_logic * logic_flow_score +
            w_contradiction * contradiction_score
        )
        
        # Ensure result is in valid range
        final_score = max(0.0, min(1.0, final_score))
        
        logger.info(
            f"Weighted score calculation: {final_score:.4f} = "
            f"({w_similarity:.2f} × {similarity_score:.3f}) + "
            f"({w_logic:.2f} × {logic_flow_score:.3f}) + "
            f"({w_contradiction:.2f} × {contradiction_score:.3f})"
        )
        
        return {
            'similarity_score': similarity_score,
            'logic_flow_score': logic_flow_score,
            'contradiction_score': contradiction_score,
            'final_score': float(final_score)
        }
        
    except Exception as e:
        logger.error(f"Final score calculation failed: {str(e)}")
        # Return average of inputs as fallback
        fallback_score = float((similarity_score + logic_flow_score + contradiction_score) / 3)
        return {
            'similarity_score': similarity_score,
            'logic_flow_score': logic_flow_score,
            'contradiction_score': contradiction_score,
            'final_score': fallback_score
        }


def get_grade_letter(score: float) -> str:
    """
    Convert numerical score to letter grade
    
    Args:
        score: Final score (0-1)
        
    Returns:
        Letter grade (A+, A, B+, etc.)
    """
    if score >= 0.95:
        return "A+"
    elif score >= 0.90:
        return "A"
    elif score >= 0.85:
        return "A-"
    elif score >= 0.80:
        return "B+"
    elif score >= 0.75:
        return "B"
    elif score >= 0.70:
        return "B-"
    elif score >= 0.65:
        return "C+"
    elif score >= 0.60:
        return "C"
    elif score >= 0.55:
        return "C-"
    elif score >= 0.50:
        return "D"
    else:
        return "F"


def get_verdict_level(score: float) -> str:
    """
    Get verdict level for frontend display
    
    Args:
        score: Final score (0-1)
        
    Returns:
        Verdict level: 'excellent', 'good', 'needs-improvement', 'poor'
    """
    if score >= 0.85:
        return "excellent"
    elif score >= 0.70:
        return "good"
    elif score >= 0.50:
        return "needs-improvement"
    else:
        return "poor"


def calculate_percentage(score: float) -> float:
    """
    Convert score to percentage
    """
    return round(score * 100, 2)


def is_passing(score: float) -> bool:
    """
    Check if score meets passing threshold
    """
    pass_threshold = config['scoring']['thresholds']['pass_score']
    return score >= pass_threshold


def generate_feedback(
    final_score: float,
    similarity_score: float,
    logic_flow_score: float,
    contradiction_score: float,
    relevance_flag: str
) -> dict:
    """
    Generate detailed feedback for student
    
    Returns:
        Dict with feedback messages and improvement suggestions
    """
    feedback = {
        "overall": "",
        "strengths": [],
        "improvements": [],
        "grade": get_grade_letter(final_score),
        "percentage": calculate_percentage(final_score),
        "passed": is_passing(final_score)
    }
    
    # Overall feedback
    verdict = get_verdict_level(final_score)
    if verdict == "excellent":
        feedback["overall"] = "Excellent answer! Your response demonstrates strong understanding and is well-structured."
    elif verdict == "good":
        feedback["overall"] = "Good answer! Your response shows understanding with minor areas for improvement."
    elif verdict == "needs-improvement":
        feedback["overall"] = "Your answer needs improvement. Review the reference material and focus on key concepts."
    else:
        feedback["overall"] = "Your answer requires significant improvement. Please revisit the material and focus on understanding core concepts."
    
    # Identify strengths
    if similarity_score >= 0.8:
        feedback["strengths"].append("Strong semantic understanding of the topic")
    if logic_flow_score >= 0.8:
        feedback["strengths"].append("Well-organized and coherent response")
    if contradiction_score >= 0.9:
        feedback["strengths"].append("Factually accurate with no contradictions")
    if relevance_flag == "relevant":
        feedback["strengths"].append("Directly addresses the question")
    
    # Suggest improvements
    if similarity_score < 0.6:
        feedback["improvements"].append("Improve alignment with reference concepts")
    if logic_flow_score < 0.6:
        feedback["improvements"].append("Enhance logical structure and use transition words")
    if contradiction_score < 0.7:
        feedback["improvements"].append("Review for factual accuracy and eliminate contradictions")
    if relevance_flag == "irrelevant":
        feedback["improvements"].append("Ensure answer directly addresses the question asked")
    
    return feedback


def calculate_component_percentages(
    similarity_score: float,
    logic_flow_score: float,
    contradiction_score: float
) -> dict:
    """
    Calculate percentage scores for each component
    """
    return {
        "similarity_percentage": calculate_percentage(similarity_score),
        "logic_flow_percentage": calculate_percentage(logic_flow_score),
        "contradiction_percentage": calculate_percentage(contradiction_score)
    }
