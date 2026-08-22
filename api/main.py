"""
FastAPI app for the retail agent.

This module:
- Sets up the FastAPI application.
- Initializes the agent with the configured LLM provider.
- Exposes a /chat endpoint.
"""

from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse
from app.agent import RetailAgent
from app.llm_providers import (
    MockLLMProvider,
    BedrockLLMProvider,
    VertexLLMProvider,
    OpenAILLMProvider,
    DeepSeekLLMProvider,
)
from app.config import config
from app.retriever import StoreRetriever
from app.graph import build_graph
from langfuse import get_client
from langfuse.langchain import CallbackHandler
import os

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")

# Initialize Langfuse client
langfuse = get_client()

app = FastAPI(
    title="Retail Agent API",
    description="AI assistant for a Pakistani clothing store in Germany",
    version="0.1.0",
)


def create_agent() -> RetailAgent:
    """
    Create the agent with the configured LLM provider.
    """
    retriever = StoreRetriever(
        products_path=config.PRODUCTS_PATH,
        policies_dir=config.POLICIES_DIR,
        qdrant_url=config.QDRANT_URL,
        qdrant_api_key=config.QDRANT_API_KEY,
    )

    provider_type = config.LLM_PROVIDER.lower()

    if provider_type == "mock":
        llm = MockLLMProvider()
    elif provider_type == "openai":
        llm = OpenAILLMProvider(api_key=os.getenv("OPENAI_API_KEY"))
    elif provider_type == "deepseek":
        llm = DeepSeekLLMProvider(api_key=os.getenv("DEEPSEEK_API_KEY"))
    elif provider_type == "bedrock":
        llm = BedrockLLMProvider(model_id=config.BEDROCK_MODEL_ID, region_name=config.BEDROCK_REGION)
    elif provider_type == "vertex":
        llm = VertexLLMProvider(
            model_name=config.VERTEX_MODEL_NAME,
            project_id=config.VERTEX_PROJECT_ID,
            location=config.VERTEX_LOCATION,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider_type}")

    return RetailAgent(llm, retriever=retriever)


# Create agent at startup (passes Qdrant URL/API key from .env)
agent = create_agent()

compiled_graph = build_graph(agent)

# Credentials/host come from LANGFUSE_* env vars via get_client()
langfuse_handler = CallbackHandler(public_key=LANGFUSE_PUBLIC_KEY)

@app.post("/chat")
def chat(request: ChatRequest):
    initial_state = {"query": request.query}
    result = compiled_graph.invoke(
    initial_state,
    config={"callbacks": [langfuse_handler]},
)
    return {
        "answer": result["answer"],
        "sources": [c["id"] for c in result["context_chunks"]],
        "tool_info": result["tool_info"],
    }

@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "provider": config.LLM_PROVIDER}