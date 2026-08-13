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
from app.llm_providers import MockLLMProvider, BedrockLLMProvider, VertexLLMProvider
from app.config import config
from retriever import StoreRetriever

app = FastAPI(
    title="Retail Agent API",
    description="AI assistant for a Pakistani clothing store in Germany",
    version="0.1.0",
)


def create_agent() -> RetailAgent:
    """
    Create the agent with the configured LLM provider.
    """
    # Initialize retriever
    retriever = StoreRetriever(
        products_path=config.PRODUCTS_PATH,
        policies_dir=config.POLICIES_DIR,
    )

    # Initialize LLM provider based on config
    provider_type = config.LLM_PROVIDER.lower()

    if provider_type == "mock":
        llm = MockLLMProvider()
    elif provider_type == "bedrock":
        llm = BedrockLLMProvider(
            model_id=config.BEDROCK_MODEL_ID,
            region_name=config.BEDROCK_REGION,
        )
    elif provider_type == "vertex":
        llm = VertexLLMProvider(
            model_name=config.VERTEX_MODEL_NAME,
            project_id=config.VERTEX_PROJECT_ID,
            location=config.VERTEX_LOCATION,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider_type}")

    return RetailAgent(llm, retriever=retriever)


# Create agent at startup
agent = create_agent()


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """
    Chat with the retail agent.

    Returns an answer with sources used for grounding.
    """
    result = agent.handle_query(req.query)
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
    )


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "provider": config.LLM_PROVIDER}