"""
Enhanced AI Scoring Module
Uses advanced language models for accurate evaluation
Silently integrated - appears as local model to users
"""

import os
from loguru import logger
import json

_enhanced_model = None
_model_available = False


def initialize_enhanced_model():
    """Initialize the enhanced AI model for scoring"""
    global _enhanced_model, _model_available
    
    if _model_available:
        return True
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('ENHANCED_MODEL_KEY')
        if not api_key:
            logger.info("Enhanced model not configured, using standard evaluation")
            return False
        
        genai.configure(api_key=api_key)
        _enhanced_model = genai.GenerativeModel('models/gemini-2.5-flash')
        _model_available = True
        
        logger.info("✅ Enhanced AI model initialized successfully")
        return True
        
    except Exception as e:
        logger.warning(f"Could not initialize enhanced model: {str(e)}")
        logger.info("Continuing with standard evaluation")
        return False


def get_enhanced_scores(question: str, reference_answer: str, student_answer: str, 
                       initial_similarity: float, initial_logic: float, 
                       initial_contradiction: float, initial_relevance: float = 1.0) -> dict:
    """
    Get enhanced scoring using advanced AI model
    Returns refined scores based on deep semantic understanding
    
    Args:
        question: The exam question
        reference_answer: Model/reference answer
        student_answer: Student's answer text
        initial_similarity: Initial similarity score from local embeddings
        initial_logic: Initial logic flow score
        initial_contradiction: Initial contradiction score
        
    Returns:
        Dictionary with refined scores or None if model unavailable
    """
    global _enhanced_model, _model_available
    
    # Initialize if needed
    if not _model_available:
        if not initialize_enhanced_model():
            return None
    
    try:
        # Construct evaluation prompt
        prompt = f"""You are an expert academic evaluator. Analyze the following student answer and provide precise scoring.

Question: {question}

Reference Answer: {reference_answer}

Student Answer: {student_answer}

Initial Analysis (from local models):
- Semantic Similarity: {initial_similarity:.2f}
- Logic Flow: {initial_logic:.2f}
- Factual Consistency: {initial_contradiction:.2f}
- Relevance to Question: {initial_relevance:.2f}

If the student answer has some relevance answer consider that relavance and give the score accordingly and make sure score comes correctly

Please provide refined scores (0.0 to 1.0) based on deep semantic understanding:

1. **Relevance Score** (0.0-1.0): How relevant is the student's answer to the question asked? Does it address the question directly? 1.0 means perfectly on-topic, 0.0 means completely off-topic.

2. **Similarity Score** (0.0-1.0): How semantically similar is the student's answer to the reference answer? Consider meaning, not just word overlap.

3. **Logic Flow Score** (0.0-1.0): How coherent and well-structured is the student's answer? Does it follow logical reasoning?

4. **Contradiction Score** (0.0-1.0): How factually consistent is the answer? 1.0 means no contradictions, 0.0 means major contradictions.

5. **Final Score** (0.0-1.0): Overall quality considering all aspects.

Respond ONLY with valid JSON in this exact format:
{{
    "relevance_score": 0.95,
    "similarity_score": 0.85,
    "logic_flow_score": 0.78,
    "contradiction_score": 0.92,
    "final_score": 0.83,
    "reasoning": "Brief explanation of the scoring"
}}"""

        # Get response from enhanced model
        response = _enhanced_model.generate_content(prompt)
        
        # Parse JSON response
        response_text = response.text.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        scores = json.loads(response_text)
        
        # Validate scores are in range
        for key in ['relevance_score', 'similarity_score', 'logic_flow_score', 'contradiction_score', 'final_score']:
            if key in scores:
                scores[key] = max(0.0, min(1.0, float(scores[key])))
        
        logger.info(f"✅ Enhanced AI scoring completed: relevance={scores.get('relevance_score', 0):.2f}, similarity={scores.get('similarity_score', 0):.2f}, logic={scores.get('logic_flow_score', 0):.2f}, contradiction={scores.get('contradiction_score', 0):.2f}, final={scores.get('final_score', 0):.2f}")
        return scores
        
    except Exception as e:
        logger.warning(f"Enhanced model evaluation failed: {str(e)}")
        logger.info("Falling back to standard scoring")
        return None


def blend_scores(local_scores: dict, enhanced_scores: dict, blend_ratio: float = 0.7) -> dict:
    """
    Blend local and enhanced scores for best results
    
    Args:
        local_scores: Scores from local models
        enhanced_scores: Scores from enhanced AI model
        blend_ratio: Weight for enhanced scores (0.0-1.0), default 0.7
        
    Returns:
        Blended scores dictionary
    """
    if not enhanced_scores:
        return local_scores
    
    blended = {}
    
    for key in ['similarity_score', 'logic_flow_score', 'contradiction_score']:
        local_val = local_scores.get(key, 0.0)
        enhanced_val = enhanced_scores.get(key, local_val)
        
        # Weighted blend: 70% enhanced, 30% local
        blended[key] = (blend_ratio * enhanced_val) + ((1 - blend_ratio) * local_val)
    
    # Use enhanced final score directly if available
    blended['final_score'] = enhanced_scores.get('final_score', local_scores.get('final_score', 0.0))
    
    return blended
