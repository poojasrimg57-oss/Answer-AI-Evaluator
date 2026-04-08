"""
Step 4: NLP Text Preprocessing Module
Cleans and preprocesses text for semantic analysis
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
from loguru import logger
import yaml

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) if config['nlp']['remove_stopwords'] else set()

# Load spaCy model (optional but recommended)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None


def clean_text(text: str) -> str:
    """
    Basic text cleaning - removes noise, normalizes whitespace
    
    Args:
        text: Raw text from OCR
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special OCR artifacts
    text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
    
    # Fix common OCR errors
    text = text.replace('|', 'I')
    text = text.replace('0', 'O').replace('l', 'I')  # Only in specific contexts
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Normalize punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    
    logger.debug(f"Cleaned text length: {len(text)} characters")
    
    return text


def preprocess_text(text: str) -> list:
    """
    Advanced text preprocessing with tokenization and lemmatization
    
    Steps:
    1. Lowercase (optional)
    2. Tokenization
    3. Remove stopwords (optional)
    4. Lemmatization
    5. Filter short words
    
    Args:
        text: Cleaned text
        
    Returns:
        List of preprocessed tokens
    """
    if not text:
        return []
    
    # Lowercase
    if config['nlp']['lowercase']:
        text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stopwords
    if config['nlp']['remove_stopwords']:
        tokens = [token for token in tokens if token.lower() not in stop_words]
    
    # Lemmatization
    if config['nlp']['lemmatization']:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Filter short words
    min_length = config['nlp']['min_word_length']
    tokens = [token for token in tokens if len(token) >= min_length]
    
    logger.debug(f"Preprocessed to {len(tokens)} tokens")
    
    return tokens


def preprocess_with_spacy(text: str) -> dict:
    """
    Advanced preprocessing using spaCy (better than NLTK)
    
    Returns:
        Dict with tokens, lemmas, POS tags, entities
    """
    if nlp is None:
        logger.warning("spaCy not available, using NLTK fallback")
        tokens = preprocess_text(text)
        return {
            "tokens": tokens,
            "lemmas": tokens,
            "pos_tags": [],
            "entities": []
        }
    
    doc = nlp(text)
    
    # Extract tokens and lemmas
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    lemmas = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
    
    # Remove stopwords if configured
    if config['nlp']['remove_stopwords']:
        filtered_tokens = []
        filtered_lemmas = []
        for token, lemma in zip(tokens, lemmas):
            if token.lower() not in stop_words:
                filtered_tokens.append(token)
                filtered_lemmas.append(lemma)
        tokens = filtered_tokens
        lemmas = filtered_lemmas
    
    # POS tags
    pos_tags = [(token.text, token.pos_) for token in doc]
    
    # Named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "entities": entities
    }


def extract_sentences(text: str) -> list:
    """
    Split text into sentences
    """
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > config['semantic_analysis']['min_sentence_length']]


def normalize_text(text: str) -> str:
    """
    Complete text normalization pipeline
    
    Returns clean, normalized text suitable for embedding generation
    """
    # Clean text
    cleaned = clean_text(text)
    
    # Preprocess with spaCy
    processed = preprocess_with_spacy(cleaned)
    
    # Rejoin lemmas to form normalized text
    normalized = " ".join(processed['lemmas'])
    
    return normalized


def remove_common_phrases(text: str) -> str:
    """
    Remove common filler phrases that don't add semantic value
    """
    filler_phrases = [
        "in my opinion",
        "I think",
        "I believe",
        "according to me",
        "in conclusion",
        "to summarize"
    ]
    
    for phrase in filler_phrases:
        text = re.sub(phrase, '', text, flags=re.IGNORECASE)
    
    return text


def detect_language(text: str) -> str:
    """
    Detect language of text (basic check)
    """
    try:
        from langdetect import detect
        return detect(text)
    except:
        # Assume English if detection fails
        return "en"
