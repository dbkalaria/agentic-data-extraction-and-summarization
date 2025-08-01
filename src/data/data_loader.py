#!/usr/bin/env python
"""
data_loader.py
----------------

Provides a standardized function to load the XSum training data from GCS,
and consistently sample it for the project.

This ensures all other modules use the exact same subset of data for
processing, evaluation, and comparison.

Usage (as a module):
--------------------
from data_loader import load_and_sample_data

df_sample = load_and_sample_data(n_samples=1000)
print(df_sample.head())
"""

import io

import pandas as pd
from google.cloud import storage
from google.api_core import exceptions

from core.config import settings
from core.logging_config import logger

def download_blob_as_string(bucket_name: str, blob_name: str) -> str:
    """Downloads a blob from GCS and returns it as a decoded string."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        logger.info(f"Downloading gs://{bucket_name}/{blob_name} from GCS...")
        data = blob.download_as_bytes().decode("utf-8")
        logger.info("Download complete.")
        return data

    except exceptions.NotFound:
        logger.error(f"Blob '{blob_name}' not found in bucket '{bucket_name}'.")
        raise
    except Exception as e:
        logger.error(f"An error occurred during download: {e}")
        raise

def load_and_sample_data(
    n_samples: int,
    max_words: int | None = None,
    text_column: str = "document",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load the XSum training split from GCS, optionally filter rows whose
    `text_column` contains fewer than `max_words` words, and return exactly
    `n_samples` of those rows.

    Raises:
        ValueError if the filtered pool has fewer than `n_samples` rows.
    """
    gcs_bucket_name = settings.gcs_bucket_name
    blob_name = "xsum/train.jsonl"

    try:
        jsonl_data = download_blob_as_string(gcs_bucket_name, blob_name)
        logger.info("Loading full dataset into memory…")
        full_df = pd.read_json(io.StringIO(jsonl_data), lines=True)
        logger.info(f"Successfully loaded {len(full_df):,} rows.")

        if max_words is not None:
            logger.info(f"Filtering rows with < {max_words} words in '{text_column}'…")
            mask = full_df[text_column].str.split().str.len() < max_words
            full_df = full_df[mask]
            logger.info(f"{len(full_df):,} rows left after filtering.")

        if len(full_df) < n_samples:
            raise ValueError(
                f"Filtered dataset has only {len(full_df)} rows; "
                f"{n_samples} requested."
            )

        sample_df = full_df.sample(n=n_samples, random_state=random_state)
        logger.info("Sampling complete.")
        return sample_df

    except Exception as e:
        logger.error(f"Failed to load or sample the data: {e}")
        raise
