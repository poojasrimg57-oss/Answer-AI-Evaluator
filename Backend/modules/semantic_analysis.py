"""
Step 7: Semantic Analysis Module
Analyzes logic flow and coherence of the answer
"""

import numpy as np
from nltk.tokenize import sent_tokenize
from loguru import logger
from modules.embeddings import compute_similarity, generate_sentence_embeddings
from modules.text_preprocessing import extract_sentences
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def analyze_logic_flow(
    student_text: str,
    reference_text: str,
    student_embedding: np.ndarray = None,
    reference_embedding: np.ndarray = None
) -> float:
    """
    Analyze the logic flow and coherence of student answer
    
    Evaluates:
    1. Sentence-to-sentence coherence
    2. Presence of logical connectors
    3. Structural similarity to reference
    4. Overall argument flow
    
    Args:
        student_text: Student answer text
        reference_text: Reference answer text
        student_embedding: Pre-computed student embedding (optional)
        reference_embedding: Pre-computed reference embedding (optional)
        
    Returns:
        Logic flow score (0 to 1)
    """
    try:
        scores = []
        
        # 1. Coherence score (sentence flow)
        coherence_score = calculate_coherence(student_text)
        scores.append(coherence_score * 0.4)  # 40% weight
        
        # 2. Logical connectors score
        connectors_score = check_logical_connectors(student_text)
        scores.append(connectors_score * 0.2)  # 20% weight
        
        # 3. Structural similarity to reference
        structure_score = compare_structure(student_text, reference_text)
        scores.append(structure_score * 0.25)  # 25% weight
        
        # 4. Argument progression score
        progression_score = analyze_argument_progression(student_text)
        scores.append(progression_score * 0.15)  # 15% weight
        
        # Combine scores
        final_score = sum(scores)
        
        logger.info(
            f"Logic flow analysis: {final_score:.3f} "
            f"(coherence={coherence_score:.2f}, "
            f"connectors={connectors_score:.2f}, "
            f"structure={structure_score:.2f}, "
            f"progression={progression_score:.2f})"
        )
        
        return float(min(1.0, max(0.0, final_score)))
        
    except Exception as e:
        logger.error(f"Logic flow analysis failed: {str(e)}")
        return 0.5  # Neutral score on error


def calculate_coherence(text: str) -> float:
    """
    Calculate sentence-to-sentence coherence using embeddings
    Measures how well sentences flow together
    """
    try:
        sentences = extract_sentences(text)
        
        if len(sentences) < 2:
            return 0.8  # Single sentence gets decent score
        
        # Generate embeddings for all sentences
        sentence_embeddings = generate_sentence_embeddings(sentences)
        
        # Calculate consecutive sentence similarities
        similarities = []
        for i in range(len(sentence_embeddings) - 1):
            sim = compute_similarity(
                sentence_embeddings[i],
                sentence_embeddings[i + 1]
            )
            similarities.append(sim)
        
        # Average similarity = coherence score
        coherence = np.mean(similarities) if similarities else 0.5
        
        return float(coherence)
        
    except Exception as e:
        logger.error(f"Coherence calculation failed: {str(e)}")
        return 0.5


def check_logical_connectors(text: str) -> float:
    """
    Check for presence of logical connectors and transition words
    These indicate structured reasoning
    """
    logic_markers = config['semantic_analysis']['logic_markers']
    
    # Additional markers
    all_markers = logic_markers + [
        'firstly', 'secondly', 'finally',
        'however', 'moreover', 'furthermore',
        'in addition', 'on the other hand',
        'for example', 'for instance',
        'in conclusion', 'to sum up'
    ]
    
    text_lower = text.lower()
    found_markers = sum(1 for marker in all_markers if marker in text_lower)
    
    # Normalize score
    # 3+ markers = excellent (1.0)
    # 2 markers = good (0.8)
    # 1 marker = fair (0.6)
    # 0 markers = poor (0.4)
    
    if found_markers >= 3:
        score = 1.0
    elif found_markers == 2:
        score = 0.8
    elif found_markers == 1:
        score = 0.6
    else:
        score = 0.4
    
    logger.debug(f"Found {found_markers} logical connectors, score: {score}")
    
    return score


def compare_structure(student_text: str, reference_text: str) -> float:
    """
    Compare structural similarity between student and reference answers
    Looks at sentence count, paragraph structure, etc.
    """
    try:
        student_sentences = extract_sentences(student_text)
        reference_sentences = extract_sentences(reference_text)
        
        # Compare sentence counts (shouldn't be too different)
        student_count = len(student_sentences)
        reference_count = len(reference_sentences)
        
        if reference_count == 0:
            return 0.5
        
        count_ratio = min(student_count, reference_count) / max(student_count, reference_count)
        
        # Check if key sentences from reference are present in student answer
        if student_count > 0 and reference_count > 0:
            student_embs = generate_sentence_embeddings(student_sentences)
            reference_embs = generate_sentence_embeddings(reference_sentences)
            
            # For each reference sentence, find best match in student answer
            matches = []
            for ref_emb in reference_embs:
                best_sim = max([
                    compute_similarity(ref_emb, stu_emb)
                    for stu_emb in student_embs
                ])
                matches.append(best_sim)
            
            avg_match = np.mean(matches)
            
            # Combine count ratio and content match
            structure_score = 0.3 * count_ratio + 0.7 * avg_match
        else:
            structure_score = count_ratio
        
        return float(structure_score)
        
    except Exception as e:
        logger.error(f"Structure comparison failed: {str(e)}")
        return 0.5


def analyze_argument_progression(text: str) -> float:
    """
    Analyze if the argument progresses logically from start to finish
    First sentences should introduce, middle should develop, end should conclude
    """
    try:
        sentences = extract_sentences(text)
        
        if len(sentences) < 3:
            return 0.7  # Short answers get decent score
        
        # Split into sections
        third = len(sentences) // 3
        intro = sentences[:third] if third > 0 else [sentences[0]]
        body = sentences[third:2*third] if third > 0 else sentences[1:-1]
        conclusion = sentences[2*third:] if third > 0 else [sentences[-1]]
        
        # Generate embeddings
        intro_emb = generate_sentence_embeddings(intro)
        body_emb = generate_sentence_embeddings(body) if body else intro_emb
        conclusion_emb = generate_sentence_embeddings(conclusion)
        
        # Check progression: intro -> body -> conclusion
        # Body should be somewhat similar to intro (develops the idea)
        # Conclusion should relate back to intro (closure)
        
        if len(intro_emb) > 0 and len(body_emb) > 0:
            intro_body_sim = np.mean([
                compute_similarity(intro_emb[0], body_e)
                for body_e in body_emb
            ])
        else:
            intro_body_sim = 0.5
        
        if len(intro_emb) > 0 and len(conclusion_emb) > 0:
            intro_conclusion_sim = compute_similarity(
                np.mean(intro_emb, axis=0),
                np.mean(conclusion_emb, axis=0)
            )
        else:
            intro_conclusion_sim = 0.5
        
        # Good progression: moderate intro-body similarity, higher intro-conclusion
        progression_score = 0.5 * intro_body_sim + 0.5 * intro_conclusion_sim
        
        return float(progression_score)
        
    except Exception as e:
        logger.error(f"Argument progression analysis failed: {str(e)}")
        return 0.5


def detect_circular_reasoning(text: str) -> bool:
    """
    Detect if answer contains circular reasoning (says same thing repeatedly)
    """
    try:
        sentences = extract_sentences(text)
        
        if len(sentences) < 2:
            return False
        
        embeddings = generate_sentence_embeddings(sentences)
        
        # Check for very high similarity between non-consecutive sentences
        for i in range(len(embeddings)):
            for j in range(i + 2, len(embeddings)):
                sim = compute_similarity(embeddings[i], embeddings[j])
                if sim > 0.95:  # Nearly identical sentences
                    logger.warning("Potential circular reasoning detected")
                    return True
        
        return False
        
    except Exception as e:
        logger.error(f"Circular reasoning detection failed: {str(e)}")
        return False
