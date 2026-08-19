# Multi‑Cloud Retail Agent

Multi‑cloud agentic RAG for a Pakistani clothing store in Germany, with AWS Bedrock + GCP Vertex integrations, local BM25 and (planned) hybrid search, and an evaluation harness ready for LLM‑as‑judge and Langfuse observability.

## ✨ Overview

This project is a retail assistant that can:

- Answer **policy questions** (shipping, returns, payment) using RAG over markdown policy docs.
- Answer **inventory and product questions** using RAG plus simple tool calls over a JSON product catalog.
- Provide **recommendations** (occasion, budget, fabric, etc.) for Pakistani clothing based on user intent.

The agent is designed to run locally with a mock LLM, and later be switched to AWS Bedrock, GCP Vertex AI, or other providers with minimal code changes.

## 🧱 Architecture

**Core components**

- `app/agent.py` – `RetailAgent` orchestration: retrieval → tools → prompt → LLM.
- `app/llm_providers.py` – pluggable `LLMProvider` interface with:
  - `MockLLMProvider` for local rule‑based responses.
  - `BedrockLLMProvider` placeholder for AWS Bedrock.
  - `VertexLLMProvider` placeholder for GCP Vertex AI.
- `retriever.py` – local **BM25** retriever over products and policy docs; no cloud dependencies.
- `app/tools.py` – helpers for inventory checks and policy lookup (used by the agent).
- `api/main.py` – FastAPI app exposing a `/chat` endpoint around the agent.

**Data**

- `data/products.json` – structured product catalog (SKU, category, fabric, colors, sizes, stock, price, occasion, care).
- `data/policies/*.md` – markdown policy documents (shipping, returns, etc.).

**Cloud & infra (planned)**

- `terraform/aws_bedrock.tf` – IAM + setup for AWS Bedrock models and future Guardrails.
- `terraform/gcp_vertex.tf` – IAM + setup for GCP Vertex AI models.
- `terraform/aws_apprunner.tf`, `terraform/gcp_cloudrun.tf` – deployment scaffolding for App Runner / Cloud Run.

## 🚦 Phased Demo Plan

The repo is structured for a cost‑aware, multi‑phase demo:

1. **Phase 1 – Local hybrid RAG + Agent (DeepSeek/OpenAI)**
   - Use `MockLLMProvider` + BM25 retriever.
   - Add Qdrant vector search for hybrid retrieval over the same products/policies.
   - Use DeepSeek/OpenAI API keys for generation and LLM‑as‑judge evaluation.
   - Wire Langfuse Hobby tier for tracing.
   - Use RAGAS metrics (faithfulness, answer relevancy, context recall) over curated golden QAs.
   - Gradio UI mounted on FastAPI to showcase the agent.

2. **Phase 2 – Docker + deployment**
   - Dockerize the FastAPI + Gradio app (hybrid search + RAG + eval hooks).
   - Prepare Docker SDK deployment to Hugging Face Spaces (CPU Basic), AWS App Runner, and GCP Cloud Run.

3. **Phase 3 – Hosted models (AWS + Vertex)**
   - Swap generation and judge models to AWS Bedrock and GCP Vertex AI equivalents.
   - Keep hybrid search: BM25 local + Qdrant vectors; only the LLM changes.
   - Start integrating Bedrock/Vertex guardrails and more formal safety policies.

4. **Phase 4 – Cost‑sensitive AWS Bedrock phase**
   - Move more of the stack to AWS‑native services (Bedrock models, Bedrock Agents, optional OpenSearch/DynamoDB).
   - Focus on minimal provisioned infrastructure and on‑demand inference only.
   - Configure Bedrock Guardrails to use cheaper filters where possible.

5. **Phase 5 – Cost‑sensitive GCP Vertex phase**
   - Mirror Phase 4 on Vertex (Gemini models, Vertex Search/Matching Engine, etc.).
   - Keep hybrid search semantics and short‑lived demos to avoid idle costs.

Each phase can be spun up for a demo and torn down quickly to keep cost minimal.

## 🧪 Evaluation & Observability

The repo includes an evaluation harness:

- `eval/golden_qa.json` – curated QA pairs for the retail domain.
- `eval/rubric.yaml` – rubric for scoring answers (e.g. correctness, grounding).
- `eval/run_eval.py` – script to run the agent against the golden set and attach scores.

This is ready to be connected to:

- **LLM‑as‑judge** – using DeepSeek/OpenAI/Bedrock/Vertex models to score answers.
- **Langfuse** – tracing and scoring runs for prompt/agent observability.
- **RAGAS** – automated evaluation of RAG pipelines.

## 🚀 Getting Started (Local, Mock LLM)

Prerequisites:

- Python 3.12+
- `pip` or `uv`

Install dependencies:

```bash
pip install -r requirements.txt
# or with uv
uv pip install -r requirements.txt
```

Run the FastAPI app:

```bash
uvicorn api.main:app --reload
```

Then open:

- `http://localhost:8000/docs` – FastAPI Swagger UI for the `/chat` endpoint.
- (Optional) Gradio UI once added, e.g. at `/ui`.

Example curl:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Do you ship to Berlin?"}'
```

## ☁️ Configuring Cloud Providers (Future Phases)

Suggested environment variables:

```bash
# OpenAI / DeepSeek
export OPENAI_API_KEY="..."
export DEEPSEEK_API_KEY="..."

# AWS Bedrock
export BEDROCK_MODEL_ID="amazon.nova-lite-v1:0"
export BEDROCK_REGION="eu-central-1"
export BEDROCK_GUARDRAIL_ID="your-guardrail-id"
export BEDROCK_GUARDRAIL_VERSION="DRAFT"

# GCP Vertex
export VERTEX_MODEL_NAME="gemini-1.5-flash"
export VERTEX_PROJECT_ID="your-project-id"
export VERTEX_LOCATION="europe-west3"

# Langfuse
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

Then switch the provider in your app initialization to `BedrockLLMProvider` or `VertexLLMProvider` instead of `MockLLMProvider` when you move to those phases.

## 🧩 Tech Stack

- **Backend**: FastAPI, Pydantic, Uvicorn.
- **RAG**: rank‑bm25 for local retrieval over JSON + markdown; Qdrant planned for hybrid dense + sparse search.
- **Agent**: `RetailAgent` with pluggable LLM provider and simple tools.
- **Cloud (planned)**: AWS Bedrock, Bedrock Guardrails, GCP Vertex AI.
- **Eval & LLMOps**: Langfuse, RAGAS, LLM‑as‑judge.
- **Infra**: Terraform modules for AWS (Bedrock, App Runner) and GCP (Vertex, Cloud Run).

## 📄 License

Add your license here (e.g. MIT).
