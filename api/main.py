"""
FastAPI app for the retail agent.

This module:
- Sets up the FastAPI application.
- Initializes the agent with the configured LLM provider.
- Exposes a /chat endpoint and a Gradio UI at /ui.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import gradio as gr
from fastapi import FastAPI
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from app.agent import RetailAgent
from app.config import config
from app.graph import build_graph
from app.llm_providers import (
    BedrockLLMProvider,
    DeepSeekLLMProvider,
    MockLLMProvider,
    OpenAILLMProvider,
    VertexLLMProvider,
)
from app.retriever import StoreRetriever
from app.schemas import ChatRequest
from ui.gradio_app import build_gradio_app

LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")

# Initialize Langfuse client (uses LANGFUSE_* env vars)
langfuse = get_client()

app = FastAPI(
    title="Retail Agent API",
    description="AI assistant for a Pakistani clothing store in Germany",
    version="0.1.0",
)

# Shared retriever + cached agents (provider -> RetailAgent)
_retriever: Optional[StoreRetriever] = None
_agents: Dict[str, RetailAgent] = {}
_graphs: Dict[str, object] = {}


def get_retriever() -> StoreRetriever:
    global _retriever
    if _retriever is None:
        _retriever = StoreRetriever(
            products_path=config.PRODUCTS_PATH,
            policies_dir=config.POLICIES_DIR,
            qdrant_url=config.QDRANT_URL,
            qdrant_api_key=config.QDRANT_API_KEY,
        )
    return _retriever


def _make_llm(provider_type: str):
    provider_type = provider_type.lower()
    if provider_type == "mock":
        return MockLLMProvider()
    if provider_type == "openai":
        return OpenAILLMProvider(api_key=os.getenv("OPENAI_API_KEY"))
    if provider_type == "deepseek":
        return DeepSeekLLMProvider(api_key=os.getenv("DEEPSEEK_API_KEY"))
    if provider_type == "bedrock":
        return BedrockLLMProvider(
            model_id=config.BEDROCK_MODEL_ID,
            region_name=config.BEDROCK_REGION,
            guardrail_id=config.BEDROCK_GUARDRAIL_ID,
            guardrail_version=config.BEDROCK_GUARDRAIL_VERSION,
        )
    if provider_type == "vertex":
        return VertexLLMProvider(
            model_name=config.VERTEX_MODEL_NAME,
            project_id=config.VERTEX_PROJECT_ID,
            location=config.VERTEX_LOCATION,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {provider_type}")


def get_agent(provider: Optional[str] = None) -> RetailAgent:
    """Return a cached agent for the given (or default) provider."""
    key = (provider or config.LLM_PROVIDER).lower()
    if key not in _agents:
        _agents[key] = RetailAgent(_make_llm(key), retriever=get_retriever())
    return _agents[key]


def get_graph(provider: Optional[str] = None):
    key = (provider or config.LLM_PROVIDER).lower()
    if key not in _graphs:
        _graphs[key] = build_graph(get_agent(key))
    return _graphs[key]


# Default agent/graph for API (LLM_PROVIDER from .env)
agent = get_agent()
compiled_graph = get_graph()
langfuse_handler = CallbackHandler(public_key=LANGFUSE_PUBLIC_KEY)


def run_query(query: str, provider: Optional[str] = None) -> dict:
    """Run a question through the LangGraph agent pipeline."""
    graph = get_graph(provider)
    result = graph.invoke(
        {"query": query},
        config={
            "callbacks": [langfuse_handler],
            "configurable": {"thread_id": f"chat-{provider or config.LLM_PROVIDER}"},
        },
    )
    return {
        "answer": result["answer"],
        "sources": [c["id"] for c in result["context_chunks"]],
        "tool_info": result.get("tool_info") or "",
        "guardrail": result.get("guardrail") or {"blocked": False},
    }


@app.post("/chat")
def chat(request: ChatRequest):
    return run_query(request.query)


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "provider": config.LLM_PROVIDER}


demo = build_gradio_app(ask=run_query, default_provider=config.LLM_PROVIDER)
app = gr.mount_gradio_app(app, demo, path="/ui")
