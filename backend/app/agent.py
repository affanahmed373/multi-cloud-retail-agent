"""
Retail agent orchestration.

This module:
- Defines the RetailAgent class.
- Handles query processing, retrieval, tool calls, and LLM generation.
- Works with any LLMProvider (mock, Bedrock, Vertex).

Guardrails (scope + PII) run in LangGraph nodes via LangChain middleware — see
app/langchain_guardrails.py and app/graph.py.
"""

from typing import Dict, Any, Optional, List

from .langchain_guardrails import scoped_system_prompt
from .llm_providers import LLMProvider
from .tools import check_inventory, get_policy
from .retriever import StoreRetriever


class RetailAgent:
    """
    Agent for the Pakistani clothing store.

    Capabilities:
    - Answer policy questions using RAG over policies.
    - Answer inventory/product questions using RAG + tools.
    - Provide recommendations based on context.
    """

    def __init__(
        self,
        llm: LLMProvider,
        retriever: Optional[StoreRetriever] = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever

    def handle_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and return an answer with sources.

        Steps:
        1. Retrieve relevant context (products + policies).
        2. Optionally call tools (inventory, policy lookup).
        3. Build a prompt for the LLM.
        4. Generate and return the answer.
        """
        # 1. Retrieve context
        context_chunks: List[Dict[str, Any]] = []
        if self.retriever:
            context_chunks = self.retriever.retrieve(query, top_k=3)

        # 2. Optional tool calls (simple heuristics for now)
        tool_info = ""
        q_lower = query.lower()

        if "stock" in q_lower or "available" in q_lower or "size" in q_lower:
            size = None
            color = None
            if " m " in q_lower or "medium" in q_lower:
                size = "M"
            elif " l " in q_lower or "large" in q_lower:
                size = "L"
            elif " s " in q_lower or "small" in q_lower:
                size = "S"

            if "black" in q_lower:
                color = "black"
            elif "white" in q_lower:
                color = "white"

            inventory = check_inventory(size=size, color=color)
            if inventory:
                tool_info = (
                    f"Inventory check: found {len(inventory)} matching items. "
                    f"Examples: {', '.join([p['name'] for p in inventory[:2]])}."
                )
            else:
                tool_info = "Inventory check: no exact matches found, but similar items may be available."

        if "return" in q_lower or "refund" in q_lower:
            policy_text = get_policy("returns_policy")
            tool_info += f"\nReturns policy excerpt:\n{policy_text[:300]}..."

        if "ship" in q_lower or "delivery" in q_lower:
            policy_text = get_policy("shipping_policy")
            tool_info += f"\nShipping policy excerpt:\n{policy_text[:300]}..."

        # 3. Build prompt
        context_text = "\n\n".join(
            [f"[{c['source_type']}] {c['id']}:\n{c['text']}" for c in context_chunks]
        )

        system_prompt = scoped_system_prompt(
            "You are a helpful assistant for a small Pakistani clothing store in Germany. "
            "Answer questions politely and concisely. "
            "Use the provided context and tool info to give accurate answers. "
            "If you are not sure, say so and suggest contacting the store directly."
        )

        prompt = f"""{system_prompt}

Context:
{context_text if context_text else "(No context retrieved)"}

Tool info:
{tool_info if tool_info else "(No tool info)"}

Question: {query}

Answer:"""

        # 4. Generate answer
        answer = self.llm.generate(prompt, system_prompt=system_prompt)

        return {
            "answer": answer,
            "sources": [c["id"] for c in context_chunks],
            "context_chunks": context_chunks,
            "tool_info": tool_info,
        }
