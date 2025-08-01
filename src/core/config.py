"""
config.py
---------

This module centralizes the configuration management for the project using Pydantic.

It defines a `Settings` class that loads environment variables from a .env file
and also holds static configuration values that are used across different modules.
This approach ensures that all configurations are in one place, type-checked, and
easily accessible.

Usage (in other modules):
-------------------------
from config import settings

# Access environment variables
gcp_project = settings.gcp_project_id

# Access static variables
model_name = settings.generative_model_name
"""

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """
    A Pydantic BaseSettings class to manage all configurations.
    It automatically reads environment variables and validates their types.
    """
    gcp_project_id: str
    gcp_location: str
    gcs_bucket_name: str
    
    firestore_collection: str
    
    vector_search_index_id: str
    vector_search_endpoint_id: str
    deployed_index_id: str

    embedding_model_name: str = "text-embedding-004"
    generative_model_name: str = "gemini-2.5-flash"
    generative_pro_model_name: str = "gemini-2.5-pro"
    
    spacy_model: str = "en_core_web_sm"

    class Config:
        """
        Pydantic configuration settings.
        - `env_file`: Specifies the file to load environment variables from.
        - `env_file_encoding`: Specifies the encoding of the .env file.
        - `extra`: Allows the model to have extra fields that are not defined.
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


settings = Settings()
