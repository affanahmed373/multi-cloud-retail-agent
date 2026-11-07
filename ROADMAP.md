# Roadmap

Planned improvements. Each item lists the skill it proves and what "done" means.
Add the skill to the CV only after the item is pushed.

## 1. Hybrid search + reranking (priority: high)
Proves: hybrid search (BM25 + dense), cross-encoder reranking.
- [ ] Add a sparse BM25 vector next to the dense `all-MiniLM-L6-v2` vector in `backend/app/retriever.py`
      (Qdrant named vectors + `fastembed` `Qdrant/bm25`), query both and fuse with RRF.
- [ ] Rerank the fused top-20 with a cross-encoder (`BAAI/bge-reranker-base`) down to top-5.
- [ ] Config switch: `RETRIEVAL_MODE=dense|hybrid|hybrid_rerank`.
- [ ] Run `rag-agent-evals` for all three modes and put the score table in the README.
Done when: README shows a dense vs hybrid vs hybrid+rerank comparison.

## 2. Live deployment (priority: high)
Proves: production deployment, public demo link for the CV.
- [ ] Deploy the Gradio UI + FastAPI to Cloud Run (or AWS App Runner) with Qdrant Cloud free tier.
- [ ] Finish the `terraform/` module for that target; remove "planned" from the README only for what is really deployed.
- [ ] Rate limit and a cheap default model so the public demo cannot run up costs.
Done when: a working URL is in the README and the CV.

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

## 8. Repo hygiene
- [ ] GitHub Actions: lint + pytest on every push; move `test_agent.py` into `tests/`.
