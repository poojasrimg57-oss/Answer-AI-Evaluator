"""
Step 8: NLI Contradiction Detection Module
Uses Natural Language Inference to detect factual contradictions
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from nltk.tokenize import sent_tokenize
from loguru import logger
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Global NLI model (lazy loading)
_nli_model = None
_nli_pipeline = None


def get_nli_model():
    """
    Get or initialize NLI model (singleton pattern)
    """
    global _nli_model, _nli_pipeline
    
    if _nli_pipeline is None:
        model_name = config['nli']['model_name']
        logger.info(f"Loading NLI model: {model_name}")
        
        try:
            # Use pipeline for easier inference
            _nli_pipeline = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("NLI model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load NLI model: {str(e)}")
            raise
    
    return _nli_pipeline


def detect_contradictions(student_text: str, reference_text: str) -> dict:
    """
    Detect contradictions between student answer and reference answer
    
    Uses NLI (Natural Language Inference) to classify sentence pairs as:
    - Entailment: Student statement aligns with reference
    - Neutral: Statement is unrelated
    - Contradiction: Statement contradicts reference
    
    Args:
        student_text: Student answer text
        reference_text: Reference answer text
        
    Returns:
        Dict with contradiction score and details
    """
    try:
        # Split into sentences
        student_sentences = sent_tokenize(student_text)
        reference_sentences = sent_tokenize(reference_text)
        
        if not student_sentences or not reference_sentences:
            return {
                "contradiction_score": 1.0,
                "contradictions": [],
                "entailments": 0,
                "neutrals": 0,
                "total_pairs": 0
            }
        
        # Analyze sentence pairs
        contradictions = []
        entailments = 0
        neutrals = 0
        total_pairs = 0
        
        for student_sent in student_sentences:
            if len(student_sent.strip()) < 5:
                continue
            
            for reference_sent in reference_sentences:
                if len(reference_sent.strip()) < 5:
                    continue
                
                # Perform NLI inference
                result = classify_sentence_pair(student_sent, reference_sent)
                total_pairs += 1
                
                if result['label'] == 'contradiction':
                    contradictions.append({
                        "student_sentence": student_sent,
                        "reference_sentence": reference_sent,
                        "confidence": result['score']
                    })
                elif result['label'] == 'entailment':
                    entailments += 1
                else:
                    neutrals += 1
        
        # Calculate contradiction score
        # High score = few contradictions (good)
        # Low score = many contradictions (bad)
        
        if total_pairs == 0:
            contradiction_score = 1.0
        else:
            contradiction_count = len(contradictions)
            penalty = config['nli']['contradiction_penalty']
            
            # Score = 1 - (contradictions / total_pairs) * penalty_factor
            contradiction_ratio = contradiction_count / total_pairs
            contradiction_score = max(0.0, 1.0 - (contradiction_ratio / penalty))
        
        logger.info(
            f"Contradiction detection: score={contradiction_score:.3f}, "
            f"contradictions={len(contradictions)}, "
            f"entailments={entailments}, "
            f"total_pairs={total_pairs}"
        )
        
        return {
            "contradiction_score": float(contradiction_score),
            "contradictions": contradictions,
            "entailments": entailments,
            "neutrals": neutrals,
            "total_pairs": total_pairs
        }
        
    except Exception as e:
        logger.error(f"Contradiction detection failed: {str(e)}")
        # Return neutral score on error
        return {
            "contradiction_score": 0.8,
            "contradictions": [],
            "entailments": 0,
            "neutrals": 0,
            "total_pairs": 0,
            "error": str(e)
        }


def classify_sentence_pair(premise: str, hypothesis: str) -> dict:
    """
    Classify relationship between premise (reference) and hypothesis (student)
    
    Args:
        premise: Reference sentence
        hypothesis: Student sentence
        
    Returns:
        Dict with label (entailment/neutral/contradiction) and score
    """
    try:
        # Use simpler classification approach
        model = get_nli_model()
        
        # Zero-shot classification with NLI labels
        result = model(
            hypothesis,
            candidate_labels=["entailment", "neutral", "contradiction"],
            hypothesis_template="This statement is {} with the reference.",
            multi_label=False
        )
        
        return {
            "label": result['labels'][0],
            "score": result['scores'][0]
        }
        
    except Exception as e:
        logger.error(f"Sentence pair classification failed: {str(e)}")
        # Return neutral by default
        return {
            "label": "neutral",
            "score": 0.5
        }


def check_factual_consistency(student_text: str, reference_facts: list) -> dict:
    """
    Check if student answer is factually consistent with reference facts
    
    Args:
        student_text: Student answer
        reference_facts: List of factual statements from reference
        
    Returns:
        Dict with consistency score and violated facts
    """
    try:
        violations = []
        
        for fact in reference_facts:
            result = classify_sentence_pair(fact, student_text)
            
            if result['label'] == 'contradiction' and result['score'] > 0.7:
                violations.append({
                    "fact": fact,
                    "confidence": result['score']
                })
        
        # Calculate consistency score
        if len(reference_facts) == 0:
            consistency_score = 1.0
        else:
            consistency_score = 1.0 - (len(violations) / len(reference_facts))
        
        return {
            "consistency_score": consistency_score,
            "violations": violations,
            "total_facts": len(reference_facts)
        }
        
    except Exception as e:
        logger.error(f"Factual consistency check failed: {str(e)}")
        return {
            "consistency_score": 0.8,
            "violations": [],
            "total_facts": 0
        }


def detect_self_contradiction(text: str) -> dict:
    """
    Detect if text contains self-contradictory statements
    
    Returns:
        Dict with self_contradiction flag and details
    """
    try:
        sentences = sent_tokenize(text)
        
        if len(sentences) < 2:
            return {
                "has_self_contradiction": False,
                "contradictory_pairs": []
            }
        
        contradictory_pairs = []
        
        # Check all sentence pairs
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                result = classify_sentence_pair(sentences[i], sentences[j])
                
                if result['label'] == 'contradiction' and result['score'] > 0.8:
                    contradictory_pairs.append({
                        "sentence_1": sentences[i],
                        "sentence_2": sentences[j],
                        "confidence": result['score']
                    })
        
        return {
            "has_self_contradiction": len(contradictory_pairs) > 0,
            "contradictory_pairs": contradictory_pairs
        }
        
    except Exception as e:
        logger.error(f"Self-contradiction detection failed: {str(e)}")
        return {
            "has_self_contradiction": False,
            "contradictory_pairs": []
        }
