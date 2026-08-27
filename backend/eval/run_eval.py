"""
Evaluation runner with Langfuse experiments and LLM-as-a-judge.

Usage:
    uv run python -m backend.eval.run_eval
    uv run python -m backend.eval.run_eval --provider mock --judge-provider openai
    uv run python -m backend.eval.run_eval --sync-dataset
    uv run python -m backend.eval.run_eval --no-judge --output backend/eval/results.json

The golden Q&A set (eval/golden_qa.json) and rubric (eval/rubric.yaml) are sufficient
to start LLM-as-a-judge eval: 20 items cover policy, inventory, and recommendations,
each with reference answers and required_facts. Expand later with guardrail cases,
multilingual queries, and adversarial prompts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from langfuse import Evaluation, Langfuse, get_client
from langfuse.experiment import LocalExperimentItem

from backend.eval.judge import get_judge
from backend.eval.rubric_loader import load_rubric

load_dotenv()

DATASET_NAME = "retail-golden-qa"
DEFAULT_GOLDEN_PATH = "eval/golden_qa.json"


def load_golden_qa(path: str = DEFAULT_GOLDEN_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def golden_to_experiment_items(samples: List[Dict[str, Any]]) -> List[LocalExperimentItem]:
    items: List[LocalExperimentItem] = []
    for s in samples:
        items.append(
            {
                "input": s["question"],
                "expected_output": s["expected_answer"],
                "metadata": {
                    "id": s["id"],
                    "required_facts": s.get("required_facts", []),
                    "difficulty": s.get("difficulty"),
                    "source_type": s.get("source_type"),
                    "source_ids": s.get("source_ids", []),
                },
            }
        )
    return items


def _item_field(item: Any, name: str, default=None):
    if hasattr(item, name):
        return getattr(item, name)
    if isinstance(item, dict):
        return item.get(name, default)
    return default


def sync_dataset_to_langfuse(
    samples: List[Dict[str, Any]],
    dataset_name: str = DATASET_NAME,
) -> None:
    """Upload golden Q&A to a Langfuse dataset for UI experiments."""
    langfuse = get_client()
    langfuse.create_dataset(
        name=dataset_name,
        description="Retail agent golden Q&A (policy, inventory, recommendations)",
        metadata={"domain": "retail", "count": len(samples)},
    )
    for s in samples:
        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input={"question": s["question"]},
            expected_output={"answer": s["expected_answer"]},
            metadata={
                "id": s["id"],
                "required_facts": s.get("required_facts", []),
                "difficulty": s.get("difficulty"),
                "source_type": s.get("source_type"),
                "source_ids": s.get("source_ids", []),
            },
            id=s["id"],
        )
    langfuse.flush()
    print(f"Synced {len(samples)} items to Langfuse dataset '{dataset_name}'")


def _unwrap_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, dict):
        return output
    return {"answer": str(output), "sources": []}


def _unwrap_input(input_data: Any) -> str:
    if isinstance(input_data, str):
        return input_data
    if isinstance(input_data, dict):
        return str(input_data.get("question") or input_data.get("input") or input_data)
    return str(input_data)


def _unwrap_expected(expected: Any) -> str:
    if isinstance(expected, str):
        return expected
    if isinstance(expected, dict):
        return str(expected.get("answer") or expected.get("expected_answer") or expected)
    return str(expected or "")


def make_agent_task(agent_provider: str):
    """Task function for Langfuse run_experiment — runs full LangGraph pipeline."""

    def task(*, item, **kwargs) -> Dict[str, Any]:
        from backend.api.main import run_query

        question = _unwrap_input(_item_field(item, "input"))
        result = run_query(question, provider=agent_provider)
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "guardrail": result.get("guardrail"),
            "tool_info": result.get("tool_info"),
        }

    return task


def required_facts_evaluator(
    *,
    input,
    output,
    expected_output=None,
    metadata=None,
    **kwargs,
) -> Evaluation:
    """Cheap heuristic: fraction of required_facts keywords reflected in output."""
    meta = metadata or {}
    facts = meta.get("required_facts") or []
    out = _unwrap_output(output)
    answer = (out.get("answer") or "").lower()
    if not facts:
        return Evaluation(name="facts_coverage", value=1.0, comment="No required facts")
    hits = sum(1 for f in facts if str(f).lower().replace("_", " ") in answer or str(f).lower() in answer)
    value = hits / len(facts)
    return Evaluation(
        name="facts_coverage",
        value=round(value, 2),
        comment=f"Matched {hits}/{len(facts)} required fact tags",
        data_type="NUMERIC",
    )


def make_llm_judge_evaluator(judge_provider: Optional[str] = None, judge_model: Optional[str] = None):
    """Factory for LLM-as-a-judge evaluator (one Langfuse score per rubric dimension)."""

    def llm_judge_evaluator(
        *,
        input,
        output,
        expected_output=None,
        metadata=None,
        **kwargs,
    ) -> Union[Evaluation, List[Evaluation]]:
        judge = get_judge(provider=judge_provider, model=judge_model)
        meta = metadata or {}
        out = _unwrap_output(output)
        result = judge.evaluate(
            question=_unwrap_input(input),
            expected=_unwrap_expected(expected_output),
            predicted=out.get("answer") or "",
            required_facts=meta.get("required_facts"),
            sources=out.get("sources"),
        )

        evaluations: List[Evaluation] = []
        for dim, data in result["dimensions"].items():
            evaluations.append(
                Evaluation(
                    name=dim,
                    value=float(data["score"]),
                    comment=data.get("reason"),
                    data_type="NUMERIC",
                    metadata={"judge_provider": judge_provider or os.getenv("JUDGE_PROVIDER", "openai")},
                )
            )
        evaluations.append(
            Evaluation(
                name="judge_average",
                value=float(result.get("average_score", 0)),
                comment=result.get("summary"),
                data_type="NUMERIC",
            )
        )
        evaluations.append(
            Evaluation(
                name="overall_pass",
                value=bool(result.get("overall_pass")),
                comment=result.get("summary"),
                data_type="BOOLEAN",
            )
        )
        return evaluations

    return llm_judge_evaluator


def run_eval(
    *,
    agent_provider: str,
    judge_provider: Optional[str] = None,
    judge_model: Optional[str] = None,
    use_judge: bool = True,
    golden_path: str = DEFAULT_GOLDEN_PATH,
    experiment_name: str = "retail-agent-eval",
    run_name: Optional[str] = None,
    sync_dataset: bool = False,
    output_path: Optional[str] = None,
    max_concurrency: int = 5,
    limit: Optional[int] = None,
) -> Any:
    samples = load_golden_qa(golden_path)
    if limit is not None:
        samples = samples[:limit]
    rubric = load_rubric()

    if sync_dataset:
        sync_dataset_to_langfuse(samples)

    langfuse = get_client()
    items = golden_to_experiment_items(samples)

    evaluators = [required_facts_evaluator]
    if use_judge:
        evaluators.append(make_llm_judge_evaluator(judge_provider, judge_model))

    metadata = {
        "agent_provider": agent_provider,
        "judge_provider": judge_provider or os.getenv("JUDGE_PROVIDER", "openai"),
        "golden_count": str(len(samples)),
        "rubric_dimensions": ",".join(d["name"] for d in rubric.get("dimensions", [])),
    }

    print(
        f"Running Langfuse experiment '{experiment_name}' "
        f"on {len(items)} items (agent={agent_provider}, judge={'on' if use_judge else 'off'})"
    )

    result = langfuse.run_experiment(
        name=experiment_name,
        run_name=run_name or f"{agent_provider}-{judge_provider or 'openai'}-judge",
        description="Retail agent golden Q&A with LLM-as-a-judge rubric scoring",
        data=items,
        task=make_agent_task(agent_provider),
        evaluators=evaluators,
        max_concurrency=max_concurrency,
        metadata=metadata,
    )

    langfuse.flush()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "experiment": experiment_name,
            "agent_provider": agent_provider,
            "judge_provider": judge_provider,
            "item_count": len(items),
            "result": str(result),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"Summary written to {output_path}")

    print("Done. Open Langfuse → Experiments / Datasets to view scores and traces.")
    return result


def main() -> None:
    from backend.app.config import config

    parser = argparse.ArgumentParser(description="Run retail agent eval with Langfuse + LLM judge")
    parser.add_argument(
        "--provider",
        default=config.LLM_PROVIDER,
        help="Agent LLM provider (mock, openai, deepseek, gemini, vertex, bedrock)",
    )
    parser.add_argument(
        "--judge-provider",
        default=os.getenv("JUDGE_PROVIDER", "openai"),
        help="Judge LLM provider (openai, deepseek, gemini)",
    )
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL"), help="Optional judge model override")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-a-judge (facts heuristic only)")
    parser.add_argument("--sync-dataset", action="store_true", help="Upload golden Q&A to Langfuse dataset")
    parser.add_argument("--golden-path", default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--experiment-name", default="retail-agent-eval")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output", default=None, help="Optional local JSON summary path")
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Run only first N golden items")
    args = parser.parse_args()

    run_eval(
        agent_provider=args.provider,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        use_judge=not args.no_judge,
        golden_path=args.golden_path,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        sync_dataset=args.sync_dataset,
        output_path=args.output,
        max_concurrency=args.max_concurrency,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
