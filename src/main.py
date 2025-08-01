#!/usr/bin/env python
"""
main.py (Unified Ingestion & Evaluation Pipeline)
-------------------------------------------------

This script serves as the single, comprehensive pipeline for the project.
It processes raw documents, runs all NLP tools, and stores the complete
set of results in Firestore for both agent consumption and evaluation.

Workflow:
1.  Loads a sample of N documents from the source data.
2.  For each document, it performs the following:
    a. Runs all implemented NLP tools (summarization, extraction).
    b. Creates a semantic vector embedding of the document's content.
    c. Stores the comprehensive results (summaries, all extracted entities,
       and a GCS URI reference) in Firestore.
    d. Upserts the document's vector into a Vertex AI Vector Search Index.

This script is designed to be re-runnable. If a document ID already exists
in Firestore, it will be updated.

Usage
-----
# Before running, ensure you have created the necessary GCP resources and
# have set the required environment variables in a .env file.

python src/main.py --samples 100
"""

import argparse
from tqdm import tqdm

from data.data_loader import load_and_sample_data
from nlp.information_extraction import extract_entities_nl_api, extract_entities_vertex_ai, extract_entities_spacy
from nlp.summarization import summarize_gemini, summarize_textrank
from core.config import settings
from core.logging_config import logger
from core.connections import db, embedding_model, index_endpoint, my_index
from core.helper import format_nl_api_entities, format_spacy_entities

def run_pipeline(args):
    """Main function to orchestrate the data processing and storage."""
    
    # --- 1. Load Configuration from .env ---
    logger.info("Configuration and GCP clients initialized via src/connections.py")

    # --- 2. Load Data ---
    logger.info(f"Loading and sampling {args.samples} documents...")
    try:
        df_to_process = load_and_sample_data(n_samples=args.samples, max_words=1000)
    except Exception as e:
        logger.error(f"Pipeline failed at data loading stage: {e}")
        return

    # --- 4. Processing and Storage Loop ---
    logger.info(f"Starting pipeline for {len(df_to_process)} documents...")
    for index, row in tqdm(df_to_process.iterrows(), total=df_to_process.shape[0], desc="Processing Documents"):
        doc_id = str(row['id'])
        doc_text = row['document']
        reference_summary = row['summary']
        source_blob_name = "xsum/train.jsonl"

        # --- Run ALL NLP Tools ---
        gemini_summary = summarize_gemini(doc_text)
        textrank_summary = summarize_textrank(doc_text)
        vertex_ai_extraction = extract_entities_vertex_ai(doc_text)
        nl_api_entities = extract_entities_nl_api(doc_text)
        spacy_entities = extract_entities_spacy(doc_text)

        # --- Create Vector Embedding ---
        try:
            embedding = embedding_model.get_embeddings([doc_text])[0].values
        except Exception as e:
            logger.error(f"Could not create embedding for doc ID {doc_id}: {e}")
            continue     

        # --- Store Comprehensive Results in Firestore ---
        firestore_doc_ref = db.collection(settings.firestore_collection).document(doc_id)
        firestore_doc_ref.set({
            'gcs_uri': f"gs://{settings.gcs_bucket_name}/{source_blob_name}",
            'document_id_in_file': doc_id,
            'reference_summary': reference_summary, 
            'gemini_summary': gemini_summary,
            'textrank_summary': textrank_summary,
            'vertex_ai_extraction': vertex_ai_extraction, 
            'nl_api_entities': format_nl_api_entities(nl_api_entities),
            'spacy_entities': format_spacy_entities(spacy_entities), 
        })

        # --- Upsert Vector to Vector Search (if configured and initialized) ---
        if index_endpoint:
            try:
                datapoints = [{
                    "datapoint_id": doc_id,
                    "feature_vector": embedding
                }]
                my_index.upsert_datapoints(datapoints=datapoints)
                logger.debug(f"Successfully upserted vector for doc ID {doc_id} to Vector Search.")
            except Exception as e:
                logger.error(f"Failed to upsert to Vector Search for doc ID {doc_id}: {e}")
        else:
            logger.warning("Skipping Vector Search upsert. Endpoint not initialized or configured.")

    logger.info("Pipeline complete. Data stored in Firestore and Vector Search.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the NLP processing and storage pipeline.")
    parser.add_argument("--samples", type=int, default=10, help="Number of documents to process and ingest.")
    args = parser.parse_args()
    run_pipeline(args)