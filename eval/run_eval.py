"""
Evaluation runner for the retail agent.

This module:
- Loads the golden Q&A dataset.
- Runs the agent over all questions.
- Saves results to a JSON file for later LLM-as-a-judge scoring.

Currently uses the mock LLM provider; later you can switch to Bedrock/Vertex.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from app.agent import RetailAgent
from app.llm_providers import MockLLMProvider
from retriever import StoreRetriever


def load_golden_qa(path: str = "eval/golden_qa.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval(
    output_path: str = "eval/results_mock.json",
    use_retriever: bool = True,
) -> None:
    """
    Run evaluation over the golden Q&A dataset.

    Saves results as JSON with:
    - id
    - question
    - expected_answer
    - predicted_answer
    - retrieved_sources (if retriever is used)
    """
    # Load dataset
    samples = load_golden_qa()

    # Initialize retriever (optional)
    retriever = None
    if use_retriever:
        retriever = StoreRetriever(
            products_path="data/products.json",
            policies_dir="data/policies",
        )

    # Initialize agent with mock LLM
    llm = MockLLMProvider()
    agent = RetailAgent(llm, retriever=retriever)

    results = []
    for s in samples:
        q = s["question"]
        out = agent.handle_query(q)

        results.append({
            "id": s["id"],
            "question": s["question"],
            "expected_answer": s["expected_answer"],
            "predicted_answer": out["answer"],
            "retrieved_sources": out.get("sources", []),
            "required_facts": s["required_facts"],
            "difficulty": s["difficulty"],
        })

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Evaluation results saved to {output_path}")


if __name__ == "__main__":
    run_eval()