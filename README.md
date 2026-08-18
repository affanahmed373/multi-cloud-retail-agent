# Multi‑Cloud Retail Agent

Multi‑cloud agentic RAG for a Pakistani clothing store in Germany, with AWS Bedrock + GCP Vertex integrations, local BM25 RAG, and an evaluation harness ready for LLM‑as‑judge and Langfuse observability.[cite:2][cite:3][cite:6]

## ✨ Overview

This project is a retail assistant that can:

- Answer **policy questions** (shipping, returns, payment) using RAG over markdown policy docs.[cite:54]
- Answer **inventory and product questions** using RAG plus simple tool calls over a JSON product catalog.[cite:54][cite:5]
- Provide **recommendations** (occasion, budget, fabric, etc.) for Pakistani clothing based on user intent.[cite:5]

The agent is designed to run locally with a mock LLM, and later be switched to AWS Bedrock and GCP Vertex AI with minimal code changes.

## 🧱 Architecture

**Core components**

- `app/agent.py` – `RetailAgent` orchestration: retrieval → tools → prompt → LLM.[cite:5]
- `app/llm_providers.py` – pluggable `LLMProvider` interface with:
  - `MockLLMProvider` for local rule‑based responses.
  - `BedrockLLMProvider` placeholder for AWS Bedrock (to be implemented).
  - `VertexLLMProvider` placeholder for GCP Vertex AI.[cite:4]
- `retriever.py` – local **BM25** retriever over products and policy docs: no cloud dependencies.[cite:54]
- `app/tools.py` – helpers for inventory checks and policy lookup (used by the agent).
- `api/main.py` – FastAPI app exposing a `/chat` endpoint around the agent.[cite:2][cite:3]

**Data**

- `data/products.json` – structured product catalog (SKU, category, fabric, colors, sizes, stock, price, occasion, care).
- `data/policies/*.md` – markdown policy documents (shipping, returns, etc.).[cite:54]

**Cloud & infra (planned)**

- `terraform/aws_bedrock.tf` – IAM + setup for AWS Bedrock models and future Guardrails.
- `terraform/gcp_vertex.tf` – IAM + setup for GCP Vertex AI models.
- `terraform/aws_apprunner.tf`, `terraform/gcp_cloudrun.tf` – deployment scaffolding for App Runner / Cloud Run.[cite:2]

## 🚦 Phased Demo Plan

The repo is structured for a cost‑aware, multi‑phase demo:

1. **Phase 1 – Local RAG + Agent**
   - Use `MockLLMProvider` + BM25 retriever.
   - Optional: wrap the agent in **LangGraph** and trace runs with **Langfuse**.
   - No cloud costs; everything runs locally.

2. **Phase 2 – AWS Bedrock Models**
   - Implement `BedrockLLMProvider` using `boto3` and Bedrock’s `converse` API.
   - Switch the FastAPI app to use Bedrock (Claude / Nova) with optional Guardrails.

3. **Phase 3 – Bedrock Agents, Guardrails & OpenSearch**
   - Add Bedrock Agent orchestration and Guardrails for safety.
   - Optionally integrate OpenSearch for vector search (short‑lived, demo‑only to avoid idle cost).

4. **Phase 4 – GCP Vertex AI**
   - Implement `VertexLLMProvider` using `google-cloud-aiplatform`.
   - Expose equivalent functionality on Vertex AI (e.g. Gemini) to mirror the Bedrock setup.

Each phase can be spun up for a demo and torn down quickly to keep cost minimal.

## 🧪 Evaluation & Observability

The repo includes an evaluation harness:

- `eval/golden_qa.json` – curated QA pairs for the retail domain.
- `eval/rubric.yaml` – rubric for scoring answers (e.g. correctness, grounding).
- `eval/run_eval.py` – script to run the agent against the golden set and attach scores.[cite:2]

This is ready to be connected to:

- **LLM‑as‑judge** – using a Bedrock/Vertex model to score answers.
- **Langfuse** – tracing and scoring runs for prompt/agent observability.

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

Environment variables (suggested):

```bash
# AWS Bedrock
export BEDROCK_MODEL_ID="amazon.nova-lite-v1:0"
export BEDROCK_REGION="eu-central-1"
export BEDROCK_GUARDRAIL_ID="your-guardrail-id"
export BEDROCK_GUARDRAIL_VERSION="DRAFT"

# Langfuse
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_HOST="https://cloud.langfuse.com"
```

Then switch the provider in your app initialization to `BedrockLLMProvider` or `VertexLLMProvider` instead of `MockLLMProvider`.

## 🧩 Tech Stack

- **Backend**: FastAPI, Pydantic, Uvicorn.[cite:6]
- **RAG**: rank‑bm25 for local retrieval over JSON + markdown.[cite:54]
- **Agent**: `RetailAgent` with pluggable LLM provider and simple tools.
- **Cloud (planned)**: AWS Bedrock, Bedrock Guardrails, GCP Vertex AI.
- **Infra**: Terraform modules for AWS (Bedrock, App Runner) and GCP (Vertex, Cloud Run).[cite:2]

## 📄 License

MIT (add your license here if different).