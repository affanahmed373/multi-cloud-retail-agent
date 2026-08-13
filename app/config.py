"""
Configuration management.

This module:
- Loads environment variables.
- Provides a simple config object for the app.
- Allows switching between mock, Bedrock, and Vertex providers.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration."""

    # LLM provider selection: "mock", "bedrock", or "vertex"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")

    # AWS Bedrock settings
    BEDROCK_MODEL_ID: str = os.getenv(
        "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
    )
    BEDROCK_REGION: str = os.getenv("BEDROCK_REGION", "us-east-1")

    # GCP Vertex AI settings
    VERTEX_MODEL_NAME: str = os.getenv("VERTEX_MODEL_NAME", "gemini-1.5-flash")
    VERTEX_PROJECT_ID: Optional[str] = os.getenv("VERTEX_PROJECT_ID")
    VERTEX_LOCATION: str = os.getenv("VERTEX_LOCATION", "us-central1")

    # Data paths
    PRODUCTS_PATH: str = os.getenv("PRODUCTS_PATH", "data/products.json")
    POLICIES_DIR: str = os.getenv("POLICIES_DIR", "data/policies")


config = Config()