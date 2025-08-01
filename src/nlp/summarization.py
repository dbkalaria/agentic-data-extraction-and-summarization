#!/usr/bin/env python
"""
summarization.py
------------------

This module consolidates all summarization functionalities for the project.

Contains functions for generating summaries using:
1.  Vertex AI with a Gemini model (for abstractive, human-like summaries).
2.  The TextRank algorithm with spaCy (as a traditional, extractive baseline).
"""
from core.prompts import GEMINI_SUMMARIZATION_PROMPT
from core.logging_config import logger
from core.connections import nlp_spacy, generative_model


def summarize_gemini(text_content: str) -> str:
    """Generates a concise summary using a Gemini model via Vertex AI.""" 
    prompt = GEMINI_SUMMARIZATION_PROMPT.format(text_content=text_content)
    try:
        response = generative_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error in Gemini summarization: {e}")
        return ""

def summarize_textrank(text_content: str, limit_sentences: int = 3) -> str:
    """Generates a summary using the TextRank algorithm."""
    if not nlp_spacy:
        logger.error("spaCy model not loaded, cannot generate TextRank summary.")
        return ""
    try:
        doc = nlp_spacy(text_content)
        summary_sentences = [sent.text.strip() for sent in doc._.textrank.summary(limit_sentences=limit_sentences)]
        return " ".join(summary_sentences)
    except Exception as e:
        logger.error(f"Error in TextRank summarization: {e}")
        return ""