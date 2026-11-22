# Roadmap

Planned improvements. Each item lists the skill it proves and what "done" means.
Add the skill to the CV only after the item is pushed.

## Frontend showcase and hosting
- Frontend: **Gradio** (`frontend/ui/gradio_app.py`), kept. A Python chat UI is enough here; the TypeScript showcase is in `ai-governance-gateway`.
- Demo hosting: Hugging Face Docker Space, free CPU tier; it sleeps when idle and wakes on the next visit (section 2).
- Cloud: Google Cloud Run with minimum instances 0, so nothing is billed while idle. AWS App Runner bills provisioned memory while idle, so it comes after Cloud Run, not before.

## 1. Hybrid search + reranking (priority: high)
Proves: hybrid search (BM25 + dense), cross-encoder reranking.
- [ ] Add a sparse BM25 vector next to the dense `all-MiniLM-L6-v2` vector in `backend/app/retriever.py`
      (Qdrant named vectors + `fastembed` `Qdrant/bm25`), query both and fuse with RRF.
- [ ] Rerank the fused top-20 with a cross-encoder (`BAAI/bge-reranker-base`) down to top-5.
- [ ] Config switch: `RETRIEVAL_MODE=dense|hybrid|hybrid_rerank`.
- [ ] Run `rag-agent-evals` for all three modes and put the score table in the README.
Done when: README shows a dense vs hybrid vs hybrid+rerank comparison.

## 2. Live demo on Hugging Face Spaces (priority: highest)
Proves: a deployed agent a recruiter can try, public demo link for the CV.
- [ ] Create a Docker Space from the existing `Dockerfile` (it already serves on port 7860). Send `/` to `/ui` so the Space opens on the chat.
- [ ] Use in-memory Qdrant (`QdrantClient(":memory:")`) when `QDRANT_URL` is empty and index `data/` at startup, so the Space needs no database.
- [ ] One cheap hosted model (DeepSeek or Gemini Flash) as the default, key stored as a Space secret. Keep `mock` as the fallback and list only providers that have a key.
- [ ] Check the app starts without Langfuse keys.
- [ ] Rate limit per session and cap question length so the public demo cannot run up costs.
- [ ] Example question buttons: stock, returns, size advice, and one off-topic question the guardrail blocks.
- [ ] "Try it" link and a short GIF at the top of the README.
Done when: the Space link works on a phone and is in the README, the profile README and the CV.

Later, for the cloud claim:
- [ ] Deploy the same image to Cloud Run (minimum instances 0) with Qdrant Cloud free tier.
- [ ] Finish the `terraform/` module for that target; remove "planned" from the README only for what is really deployed.

## 3. MCP server for the catalog (priority: high)
Proves: building an MCP server, not only using MCP tools.
- [ ] Expose `search_products`, `check_stock`, `get_policy` from `backend/app/tools.py` as an MCP server (`mcp` Python SDK, FastMCP).
- [ ] Show it working from Cursor or Claude Desktop in the README (screenshot).
Done when: `uv run retail-mcp` starts the server and the tools work from an MCP client.

## 4. WhatsApp / Instagram DM channel (priority: medium)
Proves: real customer channel, human handoff.
- [ ] WhatsApp Business Cloud API webhook -> existing `/chat` endpoint.
- [ ] Handoff to the shop owner for orders or when guardrails block an answer (human-in-the-loop node in `graph.py`).
- [ ] Answers in Urdu, English and German; add those questions to `rag-agent-evals`.
Done when: a test number answers product questions and hands off orders.

## 5. Observability + caching (priority: medium)
Proves: OpenTelemetry, Langfuse, semantic caching / Redis, cost and latency numbers.
- [ ] Langfuse tracing via OpenTelemetry for every graph node (retrieval, guardrails, LLM).
- [ ] Redis semantic cache (embedding similarity threshold) in front of the LLM call.
- [ ] Record p50 latency and cost per query with and without cache in the README.
Done when: README has a trace screenshot and a cache hit-rate / latency table.

## 6. vLLM provider (priority: medium)
Proves: self-hosted LLM serving.
- [ ] Add a `vllm` entry to `provider_registry.py` using the OpenAI-compatible client with a custom `base_url`.
- [ ] `docker-compose` service running vLLM with a small model (e.g. Qwen2.5-1.5B-Instruct, or AWQ-quantised).
- [ ] Compare answer quality and latency against a hosted provider in the eval suite.
Done when: `LLM_PROVIDER=vllm` works locally with docker compose.

## 7. Visual search (priority: low)
Proves: multimodal retrieval.
- [ ] CLIP image embeddings of product photos in a second Qdrant vector; "find similar" from a customer photo.

## 8. Azure OpenAI provider (priority: medium)
Proves: Azure OpenAI in public code, so the Azure CV version links an agent that runs on Azure.
- [x] Add a `gateway` provider that routes every LLM `generate()` through `ai-governance-gateway` `POST /v1/chat` (`GOVERNANCE_GATEWAY_URL`).
- [ ] Add `azure-openai` to `provider_registry.py` and `llm_providers.py` (`AzureChatOpenAI`; endpoint, deployment name and API version from `.env`).
- [ ] Point that provider (or the gateway) at a real Azure OpenAI deployment behind the gateway. Do not add a third Terraform stack here.
- [ ] Add it to the provider comparison in `rag-agent-evals`.
Done when: `LLM_PROVIDER=azure-openai` answers the golden set and the README lists seven providers.

## 9. Size advice and returns (priority: low)
Proves: tool calling over structured shop rules, not only retrieval.
- [ ] `size_advice` tool: map Pakistani sizes (S-XL, chest and length in inches) to EU sizes from `size_guide_policy.md`, and explain stitched vs unstitched suits.
- [ ] Questions on the 14-day withdrawal right (Widerrufsrecht) answered from `returns_policy.md`, with the policy section cited.
- [ ] Add both question types to `rag-agent-evals`.

## 10. Repo hygiene
- [ ] GitHub Actions: lint + pytest on every push; move `test_agent.py` into `tests/`.
- [ ] Move `backend/eval/` into `rag-agent-evals` so the golden set and judge live in one place.
- [ ] Name: keep `multi-cloud-retail-agent` once at least one cloud deployment is live and Azure OpenAI is added, so the name is true. Otherwise rename to `clothing-store-agent` (GitHub redirects the old URL; never reuse the old name afterwards).
