#!/usr/bin/env python
"""
data_ingestion.py
-----------------

Downloads the XSum dataset, streams each split directly to a
Google Cloud Storage (GCS) bucket as a JSON Lines (.jsonl) file.

Reads the GCS_BUCKET_NAME from a .env file in the project root.

Usage
-----
python src/data_ingestion.py \
    --cache_dir  ~/.cache/huggingface

Arguments
---------
--cache_dir    Where Hugging Face caches downloads (default: ~/.cache/huggingface)
"""

import argparse
import io
from pathlib import Path

from datasets import load_dataset, DatasetDict
from google.cloud import storage
from google.api_core import exceptions

from core.config import settings
from core.logging_config import logger

def fetch_xsum(cache_dir: Path) -> DatasetDict:
    """
    Load the XSum dataset from Hugging Face.
    """
    logger.info("Downloading Xsum dataset from Hugging Face.")
    return load_dataset(
        "xsum",
        trust_remote_code=True,
        cache_dir=str(cache_dir),
    )

def stream_to_gcs(bucket_name: str, blob_name: str, data: bytes):
    """
    Uploads data from a bytes object directly to a GCS blob.
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        logger.info(f"Streaming data to gs://{bucket_name}/{blob_name}")
        blob.upload_from_string(data, content_type="application/jsonl")
        logger.info(f"Successfully uploaded to {blob_name}.")

    except exceptions.NotFound:
        logger.error(f"Bucket '{bucket_name}' not found. Please check the name and permissions.")
        raise
    except Exception as e:
        logger.error(f"An error occurred during upload: {e}")
        raise

def process_and_upload_splits(ds: DatasetDict, bucket_name: str) -> None:
    """
    Serializes dataset splits to JSONL bytes in memory and uploads them to GCS.
    """
    for split in ds:
        destination_blob = f"xsum/{split}.jsonl"
        logger.info(f"Processing '{split}' split for upload.")

        with io.BytesIO() as bytes_buffer:
            ds[split].to_json(bytes_buffer, orient="records", lines=True)
            jsonl_data_bytes = bytes_buffer.getvalue()

        stream_to_gcs(bucket_name, destination_blob, jsonl_data_bytes)

def main() -> None:
    gcs_bucket_name = settings.gcs_bucket_name

    logger.info(f"Target GCS Bucket: {gcs_bucket_name}")

    parser = argparse.ArgumentParser(description="Download XSum and upload to GCS.")
    parser.add_argument("--cache_dir", default=Path.home() / ".cache/huggingface", type=Path, help="Hugging Face cache directory.")
    args = parser.parse_args()

    try:
        xsum_dataset = fetch_xsum(args.cache_dir)

        logger.info(f"Splits found: {list(xsum_dataset.keys())}")
        for split in xsum_dataset:
            logger.info(f"- {split}: {len(xsum_dataset[split]):,} rows")

        process_and_upload_splits(xsum_dataset, gcs_bucket_name)

        logger.info("All splits have been successfully uploaded to GCS.")

    except Exception as e:
        logger.error(f"The script failed with an error: {e}")

if __name__ == "__main__":
    main()
