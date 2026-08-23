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
    # Frankfurt — nearest AWS region to Duisburg
    BEDROCK_REGION: str = os.getenv("BEDROCK_REGION", "eu-central-1")
    BEDROCK_GUARDRAIL_ID: Optional[str] = os.getenv("BEDROCK_GUARDRAIL_ID")
    BEDROCK_GUARDRAIL_VERSION: Optional[str] = os.getenv(
        "BEDROCK_GUARDRAIL_VERSION", "DRAFT"
    )

    # GCP Vertex AI settings
    VERTEX_MODEL_NAME: str = os.getenv("VERTEX_MODEL_NAME", "gemini-1.5-flash")
    VERTEX_PROJECT_ID: Optional[str] = os.getenv("VERTEX_PROJECT_ID")
    # Frankfurt — nearest GCP region to Duisburg
    VERTEX_LOCATION: str = os.getenv("VERTEX_LOCATION", "europe-west3")

    # Data paths
    PRODUCTS_PATH: str = os.getenv("PRODUCTS_PATH", "data/products.json")
    POLICIES_DIR: str = os.getenv("POLICIES_DIR", "data/policies")

    QDRANT_URL: Optional[str] = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")

    # LangChain PIIMiddleware strategies: block | redact | mask | hash
    PII_INPUT_STRATEGY: str = os.getenv("PII_INPUT_STRATEGY", "redact")
    PII_OUTPUT_STRATEGY: str = os.getenv("PII_OUTPUT_STRATEGY", "redact")
    PII_CREDIT_CARD_STRATEGY: str = os.getenv("PII_CREDIT_CARD_STRATEGY", "mask")


config = Config()