"""
Similarity Threshold Calibration Module
Helps calibrate similarity thresholds when switching between different embedding models
or when migrating from external to local models.
"""

import numpy as np
from loguru import logger
from typing import List, Tuple, Dict
from modules.embeddings import generate_embeddings, compute_similarity


def calibrate_similarity_thresholds(
    reference_pairs: List[Tuple[str, str, float]],
    current_thresholds: Dict[str, float],
    num_samples: int = None
) -> Dict[str, float]:
    """
    Calibrate similarity thresholds based on reference answer pairs.
    
    This function helps adjust thresholds when:
    - Switching from remote to local SBERT models
    - Migrating between different embedding models
    - Fine-tuning scoring criteria
    
    Args:
        reference_pairs: List of (text1, text2, expected_similarity) tuples
                        where expected_similarity is the human-judged similarity (0-1)
        current_thresholds: Current threshold configuration dict with keys:
                           'relevance', 'high_similarity', 'medium_similarity', 'low_similarity'
        num_samples: Number of samples to use for calibration (None = use all)
        
    Returns:
        Dict with calibrated thresholds and calibration statistics
        
    Example:
        ```python
        reference_pairs = [
            ("Paris is the capital of France", "France's capital is Paris", 0.95),
            ("The sky is blue", "Water is wet", 0.1),
            ("Photosynthesis produces oxygen", "Plants create oxygen via photosynthesis", 0.85)
        ]
        
        current_thresholds = {
            'relevance': 0.5,
            'high_similarity': 0.7,
            'medium_similarity': 0.5,
            'low_similarity': 0.3
        }
        
        calibrated = calibrate_similarity_thresholds(reference_pairs, current_thresholds)
        ```
    """
    logger.info(f"Starting similarity threshold calibration with {len(reference_pairs)} reference pairs")
    
    if num_samples is not None and num_samples < len(reference_pairs):
        # Randomly sample
        indices = np.random.choice(len(reference_pairs), num_samples, replace=False)
        reference_pairs = [reference_pairs[i] for i in indices]
        logger.info(f"Using {num_samples} samples for calibration")
    
    # Compute similarities for all pairs
    computed_similarities = []
    expected_similarities = []
    
    for text1, text2, expected_sim in reference_pairs:
        # Generate embeddings
        emb1 = generate_embeddings(text1)
        emb2 = generate_embeddings(text2)
        
        # Compute similarity
        computed_sim = compute_similarity(emb1, emb2)
        
        computed_similarities.append(computed_sim)
        expected_similarities.append(expected_sim)
    
    computed_similarities = np.array(computed_similarities)
    expected_similarities = np.array(expected_similarities)
    
    # Calculate calibration statistics
    correlation = np.corrcoef(computed_similarities, expected_similarities)[0, 1]
    mae = np.mean(np.abs(computed_similarities - expected_similarities))
    rmse = np.sqrt(np.mean((computed_similarities - expected_similarities) ** 2))
    
    logger.info(f"Calibration statistics:")
    logger.info(f"  Correlation: {correlation:.3f}")
    logger.info(f"  MAE: {mae:.3f}")
    logger.info(f"  RMSE: {rmse:.3f}")
    
    # Find optimal threshold adjustments
    # Group pairs by expected similarity ranges
    high_sim_pairs = expected_similarities >= 0.7
    med_sim_pairs = (expected_similarities >= 0.4) & (expected_similarities < 0.7)
    low_sim_pairs = expected_similarities < 0.4
    
    calibrated_thresholds = current_thresholds.copy()
    
    # Adjust thresholds based on median computed similarities
    if np.any(high_sim_pairs):
        high_median = np.median(computed_similarities[high_sim_pairs])
        calibrated_thresholds['high_similarity'] = max(0.6, high_median - 0.1)
    
    if np.any(med_sim_pairs):
        med_median = np.median(computed_similarities[med_sim_pairs])
        calibrated_thresholds['medium_similarity'] = max(0.4, med_median - 0.05)
    
    if np.any(low_sim_pairs):
        low_median = np.median(computed_similarities[low_sim_pairs])
        calibrated_thresholds['low_similarity'] = min(0.4, low_median + 0.05)
    
    # Relevance threshold (midpoint between low and medium)
    calibrated_thresholds['relevance'] = (
        calibrated_thresholds['low_similarity'] + 
        calibrated_thresholds['medium_similarity']
    ) / 2
    
    # Ensure thresholds are ordered correctly
    calibrated_thresholds['low_similarity'] = min(
        calibrated_thresholds['low_similarity'],
        calibrated_thresholds['medium_similarity'] - 0.1
    )
    calibrated_thresholds['medium_similarity'] = min(
        calibrated_thresholds['medium_similarity'],
        calibrated_thresholds['high_similarity'] - 0.1
    )
    
    logger.info("Calibrated thresholds:")
    for key, value in calibrated_thresholds.items():
        old_value = current_thresholds.get(key, 'N/A')
        change = f"({value - old_value:+.3f})" if isinstance(old_value, float) else ""
        logger.info(f"  {key}: {value:.3f} {change}")
    
    return {
        'thresholds': calibrated_thresholds,
        'statistics': {
            'correlation': float(correlation),
            'mae': float(mae),
            'rmse': float(rmse),
            'num_samples': len(reference_pairs)
        },
        'distribution': {
            'computed_mean': float(np.mean(computed_similarities)),
            'computed_std': float(np.std(computed_similarities)),
            'expected_mean': float(np.mean(expected_similarities)),
            'expected_std': float(np.std(expected_similarities))
        }
    }


def generate_sample_calibration_pairs() -> List[Tuple[str, str, float]]:
    """
    Generate sample calibration pairs for testing.
    These are common answer similarity scenarios.
    
    Returns:
        List of (text1, text2, expected_similarity) tuples
    """
    return [
        # High similarity pairs (0.8-1.0)
        (
            "Photosynthesis is the process where plants convert light energy into chemical energy",
            "Plants use photosynthesis to transform light into chemical energy",
            0.9
        ),
        (
            "Water boils at 100 degrees Celsius at sea level",
            "At sea level, water reaches boiling point at 100°C",
            0.95
        ),
        (
            "The capital of France is Paris",
            "Paris is France's capital city",
            0.95
        ),
        
        # Medium-high similarity (0.6-0.8)
        (
            "Gravity pulls objects toward Earth",
            "Objects fall down due to gravitational force",
            0.75
        ),
        (
            "Mitochondria are the powerhouse of the cell",
            "Cells generate energy in mitochondria",
            0.7
        ),
        (
            "Shakespeare wrote Romeo and Juliet",
            "The author of Romeo and Juliet is Shakespeare",
            0.8
        ),
        
        # Medium similarity (0.4-0.6)
        (
            "Dogs are loyal pets",
            "Cats are independent animals",
            0.4
        ),
        (
            "Rain falls from clouds",
            "Weather affects outdoor activities",
            0.45
        ),
        (
            "Mathematics uses numbers and formulas",
            "Science requires careful observation",
            0.5
        ),
        
        # Low similarity (0.1-0.4)
        (
            "The sky is blue during daytime",
            "Pizza is made with cheese and tomato sauce",
            0.1
        ),
        (
            "Computers process digital information",
            "Trees provide oxygen through photosynthesis",
            0.15
        ),
        (
            "History studies past events",
            "Music creates emotional experiences",
            0.2
        ),
        
        # Very low similarity (<0.1)
        (
            "Quantum mechanics describes subatomic particles",
            "Basketball is played with a round ball",
            0.05
        ),
        (
            "DNA contains genetic information",
            "Cooking requires heat and ingredients",
            0.05
        ),
    ]


def test_calibration():
    """
    Test calibration with sample pairs
    """
    logger.info("=" * 60)
    logger.info("Testing Similarity Threshold Calibration")
    logger.info("=" * 60)
    
    # Generate sample pairs
    reference_pairs = generate_sample_calibration_pairs()
    
    # Current thresholds (example from config.yaml)
    current_thresholds = {
        'relevance': 0.5,
        'high_similarity': 0.7,
        'medium_similarity': 0.5,
        'low_similarity': 0.3
    }
    
    logger.info(f"\nCurrent thresholds:")
    for key, value in current_thresholds.items():
        logger.info(f"  {key}: {value:.3f}")
    
    # Perform calibration
    result = calibrate_similarity_thresholds(reference_pairs, current_thresholds)
    
    logger.info("\n" + "=" * 60)
    logger.info("Calibration Results:")
    logger.info("=" * 60)
    
    logger.info("\nNew thresholds:")
    for key, value in result['thresholds'].items():
        logger.info(f"  {key}: {value:.3f}")
    
    logger.info("\nStatistics:")
    for key, value in result['statistics'].items():
        logger.info(f"  {key}: {value}")
    
    logger.info("\nDistribution:")
    for key, value in result['distribution'].items():
        logger.info(f"  {key}: {value:.3f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Calibration test completed!")
    logger.info("=" * 60)
    
    return result


if __name__ == "__main__":
    # Run calibration test
    test_calibration()
