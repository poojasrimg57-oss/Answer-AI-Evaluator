"""
Tests for Text Preprocessing Module
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.text_preprocessing import (
    clean_text,
    preprocess_text,
    extract_sentences,
    normalize_text
)


def test_clean_text():
    """Test basic text cleaning"""
    raw_text = "This   is   a    test.\n\nWith  multiple   spaces."
    cleaned = clean_text(raw_text)
    assert "  " not in cleaned
    assert cleaned == "This is a test. With multiple spaces."


def test_clean_text_with_artifacts():
    """Test cleaning OCR artifacts"""
    raw_text = "This is a test | with artifacts"
    cleaned = clean_text(raw_text)
    assert "|" not in cleaned or cleaned.count("I") > 0


def test_preprocess_text():
    """Test text tokenization and preprocessing"""
    text = "The quick brown fox jumps over the lazy dog."
    tokens = preprocess_text(text)
    
    assert isinstance(tokens, list)
    assert len(tokens) > 0
    # Stopwords should be removed if configured
    assert all(isinstance(token, str) for token in tokens)


def test_preprocess_text_empty():
    """Test preprocessing with empty text"""
    tokens = preprocess_text("")
    assert tokens == []


def test_extract_sentences():
    """Test sentence extraction"""
    text = "This is sentence one. This is sentence two! And a third?"
    sentences = extract_sentences(text)
    
    assert len(sentences) == 3
    assert "This is sentence one." in sentences[0]


def test_extract_sentences_short():
    """Test sentence extraction filters short sentences"""
    text = "Hi. This is a longer sentence that should be included."
    sentences = extract_sentences(text)
    
    # "Hi." might be filtered out due to min_sentence_length config
    assert len(sentences) >= 1
    assert any("longer" in s for s in sentences)


def test_normalize_text():
    """Test complete text normalization"""
    text = "The QUICK brown FOX jumps!"
    normalized = normalize_text(text)
    
    assert isinstance(normalized, str)
    assert len(normalized) > 0


def test_special_characters():
    """Test handling of special characters"""
    text = "Test with @#$% special chars & symbols"
    cleaned = clean_text(text)
    
    # Should remove most special characters except basic punctuation
    assert "&" not in cleaned or "and" in cleaned.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
