#!/usr/bin/env python
"""
preprocessing.py
------------------

This module contains functions for classic NLP text preprocessing.

These steps (e.g., lowercasing, stop-word removal) are generally required
for traditional, statistical NLP models but are often detrimental to the
performance of modern, large language models.

This module is provided to allow for comparison and to fulfill project
requirements for including traditional baselines.
"""

from core.logging_config import logger
from core.connections import nlp_spacy


def preprocess_for_traditional_nlp(text: str) -> str:
    """
    Performs classic NLP preprocessing on a text string:
    - Converts to lowercase.
    - Removes stop words and punctuation.
    - Lemmatizes tokens (reduces words to their root form).
    """
    if not nlp_spacy:
        logger.error("Cannot preprocess text, spaCy model is not loaded.")
        return text 

    doc = nlp_spacy(text.lower())
    
    processed_tokens = [ 
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct 
    ]
    
    return " ".join(processed_tokens)
