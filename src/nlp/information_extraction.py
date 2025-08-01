#!/usr/bin/env python
"""
information_extraction.py
---------------------------

This module consolidates all information extraction functionalities for the project.

Contains functions for extracting entities using:
1.  Google Cloud Natural Language API (for standard entities).
2.  Vertex AI with a Gemini model (for custom, prompt-based entities).
3.  spaCy (as a traditional, open-source baseline).
"""

import json
import re
from google.cloud import language_v1
from vertexai.generative_models import GenerationConfig

from core.prompts import ENTITY_EXTRACTION_PROMPT
from core.logging_config import logger
from core.connections import nlp_spacy, generative_model


def extract_entities_nl_api(text_content: str) -> list:
    """Analyzes standard entities using the Google Cloud Natural Language API."""
    try:
        client = language_v1.LanguageServiceClient()
        document = language_v1.Document(content=text_content, type_=language_v1.Document.Type.PLAIN_TEXT)
        response = client.analyze_entities(document=document)
        return response.entities
    except Exception as e:
        logger.error(f"Error in NL API extraction: {e}")
        return []

def extract_entities_vertex_ai(text_content: str) -> dict:
    """
    Extracts custom entities using a Gemini model's JSON mode via Vertex AI.
    This is a more robust method that significantly reduces parsing errors.
    """
    try:
        generation_config = GenerationConfig(
            response_mime_type="application/json",
        )

        prompt = ENTITY_EXTRACTION_PROMPT.format(text_content=text_content)

        response = generative_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        json_string = response.text.strip()
        
        match = re.search(r"```(json)?\s*(\{.*?\})\s*```", json_string, re.DOTALL)
        if match:
            clean_json_string = match.group(2)
        else:
            clean_json_string = json_string

        return json.loads(clean_json_string)

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from model response. Error: {e}")
        logger.error(f"Model's raw response text: {response.text}") 
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred in Vertex AI extraction: {e}")
        return {}

def extract_entities_spacy(text_content: str) -> list:
    """Extracts named entities using spaCy's NER model."""
    if not nlp_spacy:
        logger.error("spaCy model not loaded, cannot extract entities.")
        return []
    try:
        doc = nlp_spacy(text_content)
        return doc.ents
    except Exception as e:
        logger.error(f"Error in spaCy extraction: {e}")
        return []