#!/usr/bin/env python
"""
evaluate.py (Offline Evaluation Script)
---------------------------------------

This script performs offline evaluation by reading pre-processed data
directly from Firestore. It calculates ROUGE scores for summarization
and outputs all collected data to a CSV for detailed analysis.

Usage
-----
# Ensure your .env file has GCP_PROJECT_ID and FIRESTORE_COLLECTION set.
python src/evaluate.py --limit 100 --output evaluation_results.csv

Arguments
---------
--limit     Optional: Maximum number of documents to fetch from Firestore (default: all).
--output    The path to the output CSV file (default: evaluation_results.csv).
"""

import argparse
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer

from core.config import settings
from core.logging_config import logger
from core.connections import db


def run_offline_evaluation(args):
    """Main function to fetch data from Firestore and perform evaluation."""

    logger.info("Loading configuration for OFFLINE EVALUATION...")
    firestore_collection = settings.firestore_collection

    logger.info(f"Fetching documents from Firestore collection '{firestore_collection}'...")
    docs_ref = db.collection(firestore_collection)

    if args.limit:
        docs = docs_ref.limit(args.limit).stream()
    else:
        docs = docs_ref.stream()

    fetched_data = []
    for doc in tqdm(docs, desc="Fetching from Firestore"):
        data = doc.to_dict()
        data['id'] = doc.id 
        fetched_data.append(data)

    if not fetched_data:
        logger.warning(f"No documents found in Firestore collection '{firestore_collection}'. Please run main.py first.")
        return

    logger.info(f"Successfully fetched {len(fetched_data)} documents.")

    results_for_csv = []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    logger.info("Calculating ROUGE scores and formatting data...")
    for item in tqdm(fetched_data, desc="Calculating Metrics"):
        doc_id = item.get('id', 'N/A')
        reference_summary = item.get('reference_summary', '')
        gemini_summary = item.get('gemini_summary', '')
        textrank_summary = item.get('textrank_summary', '')

        # Calculate ROUGE scores
        gemini_scores = scorer.score(reference_summary, gemini_summary)
        textrank_scores = scorer.score(reference_summary, textrank_summary)

        row_data = {
            'id': doc_id,
            'document_id_in_file': item.get('document_id_in_file', ''),
            'gcs_uri': item.get('gcs_uri', ''),
            'reference_summary': reference_summary,
            'gemini_summary': gemini_summary,
            'textrank_summary': textrank_summary,
            'gemini_rougeL_fmeasure': gemini_scores['rougeL'].fmeasure,
            'textrank_rougeL_fmeasure': textrank_scores['rougeL'].fmeasure,
        }
        results_for_csv.append(row_data)

    logger.info(f"Formatting complete. Saving results to {args.output}...")
    results_df = pd.DataFrame(results_for_csv)
    results_df.to_csv(args.output, index=False)
    logger.info(f"Successfully saved evaluation results to {args.output}.")
    print(f"\nOffline evaluation complete. Results are in {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run offline evaluation from Firestore data.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of documents to fetch from Firestore.")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Output CSV file path.")
    args = parser.parse_args()
    run_offline_evaluation(args)
