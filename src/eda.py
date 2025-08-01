#!/usr/bin/env python
"""
data_processing.py
--------------------

Downloads a data split from Google Cloud Storage, loads it into a Pandas DataFrame,
and performs a basic Exploratory Data Analysis (EDA).

This script focuses on analyzing the 'validation' split for efficiency.

Usage
-----
python eda.py
"""

import io

import pandas as pd
from google.cloud import storage
from google.api_core import exceptions

from core.config import settings
from core.logging_config import logger

def download_blob_as_string(bucket_name: str, blob_name: str) -> str:
    """Downloads a blob from GCS and returns it as a string."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        logger.info(f"Downloading gs://{bucket_name}/{blob_name}...")
        data = blob.download_as_bytes().decode("utf-8")
        logger.info("Download complete.")
        return data

    except exceptions.NotFound:
        logger.error(f"Blob '{blob_name}' not found in bucket '{bucket_name}'.")
        raise
    except Exception as e:
        logger.error(f"An error occurred during download: {e}")
        raise

def perform_eda(jsonl_data: str):
    """Loads JSONL data into a DataFrame and prints length statistics."""
    logger.info("Performing Exploratory Data Analysis (EDA)...")

    df = pd.read_json(io.StringIO(jsonl_data), lines=True)

    if 'document' not in df.columns or 'summary' not in df.columns:
        logger.error("The data must contain 'document' and 'summary' columns.")
        return

    df['document_length'] = df['document'].str.split().str.len()
    df['summary_length'] = df['summary'].str.split().str.len()

    print("\n--- EDA Results ---")
    print("Dataset sample:")
    print(df.head())
    print("\nDescriptive Statistics for Text Lengths:")
    print(df[['document_length', 'summary_length']].describe())
    print("-------------------")

def main():
    """Main function to orchestrate the data processing and EDA."""
    gcs_bucket_name = settings.gcs_bucket_name
    blob_to_analyze = "xsum/validation.jsonl"

    try:
        jsonl_content = download_blob_as_string(gcs_bucket_name, blob_to_analyze)

        perform_eda(jsonl_content)

    except Exception as e:
        logger.error(f"The script failed with an error: {e}")

if __name__ == "__main__":
    main()
