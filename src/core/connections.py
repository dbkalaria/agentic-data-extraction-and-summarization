"""
connections.py
----------------

This module centralizes the initialization of external services and models.

It handles the setup for:
- Google Cloud services (Vertex AI, Firestore)
- AI Platform models (Vector Search)
- Language models (Gemini, Text Embedding)
- spaCy NLP model

By initializing these resources once, we avoid redundant connections and ensure
that all other modules use the same, consistently configured instances.

Usage (in other modules):
-------------------------
from src.connections import (
    db, generative_model, embedding_model, index_endpoint, nlp_spacy
)
"""

import spacy
import pytextrank
import vertexai
from google.cloud import aiplatform, firestore
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel

from core.config import settings
from core.logging_config import logger

# --- Initialize Google Cloud Services ---
try:
    logger.info(f"Initializing Vertex AI for project '{settings.gcp_project_id}' in '{settings.gcp_location}'")
    vertexai.init(project=settings.gcp_project_id, location=settings.gcp_location)
    db = firestore.Client(project=settings.gcp_project_id)
    logger.info("Firestore client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Google Cloud services: {e}", exc_info=True)
    db = None

# --- Initialize Language Models ---
try:
    logger.info(f"Loading Generative Model: {settings.generative_model_name}")
    generative_model = GenerativeModel(settings.generative_model_name)
    generative_pro_model = GenerativeModel(settings.generative_pro_model_name)

    logger.info(f"Loading Embedding Model: {settings.embedding_model_name}")
    embedding_model = TextEmbeddingModel.from_pretrained(settings.embedding_model_name)
except Exception as e:
    logger.error(f"Failed to load language models: {e}", exc_info=True)
    generative_model = None
    embedding_model = None

# --- Initialize Vertex AI Vector Search ---
try:
    logger.info(f"Connecting to Vector Search index: {settings.vector_search_index_id}")
    my_index = aiplatform.MatchingEngineIndex(settings.vector_search_index_id)

    logger.info(f"Connecting to Vector Search endpoint: {settings.vector_search_endpoint_id}")
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint(settings.vector_search_endpoint_id)
except Exception as e:
    logger.error(f"Failed to connect to Vertex AI Vector Search: {e}", exc_info=True)
    my_index = None
    index_endpoint = None

# --- Initialize spaCy Model ---
try:
    logger.info(f"Loading spaCy model: {settings.spacy_model}")
    nlp_spacy = spacy.load(settings.spacy_model)
    if "textrank" not in nlp_spacy.pipe_names:
        nlp_spacy.add_pipe("textrank", last=True)
except OSError:
    logger.error(f"spaCy model '{settings.spacy_model}' not found. Please run: python -m spacy download {settings.spacy_model}")
    nlp_spacy = None
except Exception as e:
    logger.error(f"An unexpected error occurred while loading the spaCy model: {e}", exc_info=True)
    nlp_spacy = None
