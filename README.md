# Multi‑Cloud Retail Agent

Multi‑cloud agentic RAG for a Pakistani clothing store in Germany. The agent answers policy and product questions using Qdrant vector search, runs through a LangGraph pipeline with LangChain guardrails (PII + scope checks), and supports OpenAI, DeepSeek, Gemini, Vertex AI, AWS Bedrock, or a local mock LLM.

## Overview

This project is a retail assistant that can:

- Answer **policy questions** (shipping, returns, payment) using RAG over markdown policy docs.
- Answer **inventory and product questions** using RAG plus simple tool calls over a JSON product catalog.
- Provide **recommendations** (occasion, budget, fabric, etc.) for Pakistani clothing based on user intent.

Switch LLM providers via `.env` or the Gradio UI without changing application code.

## Architecture

```
User → Gradio /ui  or  FastAPI /chat
         ↓
    LangGraph pipeline
         ↓
  input_guardrails  (LangChain PIIMiddleware + scope checks)
         ↓
       agent         (RetailAgent: Qdrant RAG → tools → LLM)
         ↓
  output_guardrails (PII redaction on answers)
         ↓
      response
```

**Core components**

| Module | Role |
|--------|------|
| `app/agent.py` | `RetailAgent`: retrieval → tools → prompt → LLM |
| `app/graph.py` | LangGraph pipeline with input/output guardrail nodes |
| `app/langchain_guardrails.py` | LangChain `PIIMiddleware` + custom scope middleware |
| `app/guardrails.py` | Scope/injection/off-topic policy patterns |
| `app/retriever.py` | Qdrant + `sentence-transformers` semantic search over products & policies |
| `app/llm_providers.py` | Pluggable providers (mock, OpenAI, DeepSeek, Gemini, Vertex, Bedrock) |
| `app/provider_registry.py` | Provider list, labels, and config checks for API + Gradio |
| `app/tools.py` | Inventory checks and policy lookup helpers |
| `api/main.py` | FastAPI app: `/chat`, `/health`, Gradio UI at `/ui` |
| `ui/gradio_app.py` | Chat UI with per-question provider selection |

**Data**

- `data/products.json` – product catalog (SKU, category, fabric, colors, sizes, stock, price, occasion, care).
- `data/policies/*.md` – markdown policy documents (shipping, returns, payment, size guide).

**Cloud & infra (planned)**

- `terraform/aws_bedrock.tf`, `terraform/gcp_vertex.tf` – IAM and model setup.
- `terraform/aws_apprunner.tf`, `terraform/gcp_cloudrun.tf` – deployment scaffolding.

## LLM providers

| Provider | ID | Required config |
|----------|----|-----------------|
| Mock (local rules) | `mock` | none |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` |
| Gemini (Google AI) | `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| Vertex AI (GCP Gemini) | `vertex` | `VERTEX_PROJECT_ID` + GCP credentials |
| AWS Bedrock | `bedrock` | AWS credentials; optional `BEDROCK_GUARDRAIL_ID` |

Set the default with `LLM_PROVIDER` in `.env`. The Gradio UI at `/ui` lets you pick any provider per question.

## Guardrails

Guardrails run on **every** provider via LangGraph nodes:

- **Input** – prompt injection, off-topic/unsafe requests, API key blocking, PII redaction (email, credit card, IP, URL).
- **Output** – PII redaction and secret blocking before the answer is returned.
- **Bedrock** – optional native Guardrails via `BEDROCK_GUARDRAIL_ID` when using the Bedrock provider.

Configure PII handling in `.env`:

```bash
PII_INPUT_STRATEGY=redact    # block | redact | mask | hash
PII_OUTPUT_STRATEGY=redact
PII_CREDIT_CARD_STRATEGY=mask
```

Blocked or redacted runs include `guardrail` metadata in the API response.

## Getting started

**Prerequisites**

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- Qdrant (local or [Qdrant Cloud](https://qdrant.tech/cloud/))

**Install**

```bash
git clone <repo-url>
cd multi-cloud-retail-agent
uv sync
```

**Environment**

Copy `temp/env_example.txt` to `.env` and fill in the values you need:

```bash
cp temp/env_example.txt .env
```

Minimum for local testing with mock LLM + Qdrant Cloud:

```bash
LLM_PROVIDER=mock
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

**Run**

```bash
uv run uvicorn api.main:app --reload
```

**Endpoints**

| URL | Description |
|-----|-------------|
| http://localhost:8000/ui | Gradio chat (pick provider per question) |
| http://localhost:8000/docs | Swagger UI for `/chat` |
| http://localhost:8000/health | Health check + provider readiness |
| http://localhost:8000/chat | REST API |

**Example API call**

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Do you ship to Berlin?"}'
```

**Quick agent test (no server)**

```bash
uv run python test_agent.py
```

## Environment variables

See `temp/env_example.txt` for the full list. Key groups:

```bash
# Default provider
LLM_PROVIDER=mock

# OpenAI / DeepSeek / Gemini
OPENAI_API_KEY=
DEEPSEEK_API_KEY=
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash

# AWS Bedrock
BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0
BEDROCK_REGION=eu-central-1
BEDROCK_GUARDRAIL_ID=
BEDROCK_GUARDRAIL_VERSION=DRAFT

# GCP Vertex AI
VERTEX_MODEL_NAME=gemini-1.5-flash
VERTEX_PROJECT_ID=
VERTEX_LOCATION=europe-west3

# Qdrant vector store
QDRANT_URL=
QDRANT_API_KEY=

# Langfuse tracing
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com

# PII guardrails
PII_INPUT_STRATEGY=redact
PII_OUTPUT_STRATEGY=redact
PII_CREDIT_CARD_STRATEGY=mask
```

## Phased demo plan

The repo is structured for cost‑aware, incremental demos:

1. **Phase 1 – Local RAG + multi-provider agent** *(current)*
   - Qdrant vector search, LangGraph agent, LangChain guardrails, Gradio UI.
   - OpenAI / DeepSeek / Gemini API keys for generation; Langfuse for tracing.
   - RAGAS + LLM-as-judge over `eval/golden_qa.json` (harness in place).

2. **Phase 2 – Docker + deployment**
   - Dockerize FastAPI + Gradio; deploy to HF Spaces, AWS App Runner, or GCP Cloud Run.

3. **Phase 3 – Hosted models (Bedrock + Vertex)**
   - Production inference on AWS Bedrock and GCP Vertex; hybrid search unchanged.

4. **Phase 4 / 5 – Cost-sensitive AWS and GCP phases**
   - Native cloud services, minimal provisioned infra, on-demand inference only.

## Evaluation & observability

- `eval/golden_qa.json` – curated QA pairs for the retail domain.
- `eval/rubric.yaml` – scoring rubric (correctness, groundedness, completeness, etc.).
- `eval/run_eval.py` – run the agent against the golden set.
- **Langfuse** – traces LangGraph runs via `CallbackHandler` (set `LANGFUSE_*` in `.env`).

Ready to connect **LLM-as-judge** and **RAGAS** for automated eval.

## Tech stack

- **Backend**: FastAPI, Pydantic, Uvicorn, Gradio
- **Agent**: LangGraph, LangChain middleware (`PIIMiddleware`)
- **RAG**: Qdrant, sentence-transformers (`all-MiniLM-L6-v2`), rank-bm25 (legacy/local)
- **LLMs**: OpenAI, DeepSeek, Gemini API, GCP Vertex AI, AWS Bedrock, mock
- **Observability**: Langfuse
- **Infra (planned)**: Terraform for AWS Bedrock / App Runner and GCP Vertex / Cloud Run

## License

Add your license here (e.g. MIT).
