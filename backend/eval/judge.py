"""
LLM-as-a-judge for retail agent evaluation.

Uses the rubric in eval/rubric.yaml and scores each dimension 1–5 with reasoning.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from backend.app.config import config
from backend.app.llm_providers import (
    DeepSeekLLMProvider,
    GeminiLLMProvider,
    LLMProvider,
    OpenAILLMProvider,
)
from backend.eval.rubric_loader import format_rubric_for_prompt, load_rubric


def _make_judge_llm(provider: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
    provider = (provider or config.JUDGE_PROVIDER).lower()
    model = model or config.JUDGE_MODEL

    if provider == "openai":
        return OpenAILLMProvider(
            model=model or "gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    if provider == "deepseek":
        return DeepSeekLLMProvider(
            model=model or "deepseek-chat",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )
    if provider == "gemini":
        return GeminiLLMProvider(
            model=model or config.GEMINI_MODEL,
            api_key=config.GEMINI_API_KEY,
        )
    raise ValueError(
        f"Unsupported judge provider: {provider}. Use openai, deepseek, or gemini."
    )


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


class LLMJudge:
    """Score agent answers against reference answers using a rubric."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        rubric_path: str = "eval/rubric.yaml",
    ) -> None:
        self.llm = _make_judge_llm(provider, model)
        self.rubric = load_rubric(rubric_path)
        self.dimension_names = [d["name"] for d in self.rubric.get("dimensions", [])]
        self.rubric_text = format_rubric_for_prompt(self.rubric)

    def evaluate(
        self,
        *,
        question: str,
        expected: str,
        predicted: str,
        required_facts: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        facts = required_facts or []
        src = sources or []
        context_text = ""
        if context_chunks:
            context_text = "\n".join(
                f"- [{c.get('source_type', '?')}] {c.get('id', '?')}: "
                f"{(c.get('text') or '')[:200]}"
                for c in context_chunks[:5]
            )

        prompt = f"""You are an expert evaluator for a Pakistani clothing store assistant in Germany.

Score the PREDICTED answer against the REFERENCE answer using the rubric below.
Use integer scores from 1 to 5 for each dimension.

RUBRIC:
{self.rubric_text}

EVALUATION INPUT
Question: {question}

Reference answer (ground truth):
{expected}

Predicted answer (agent output):
{predicted}

Required facts that should be reflected (semantic match OK, exact wording not required):
{json.dumps(facts, ensure_ascii=False)}

Retrieved source IDs: {json.dumps(src, ensure_ascii=False)}
Retrieved context excerpts:
{context_text or "(none provided)"}

Instructions:
- correctness: factual match to reference and store data
- groundedness: claims supported by retrieved context / known catalog
- completeness: all parts of the question addressed
- helpfulness: actionable and useful to the shopper
- tone: polite, boutique-appropriate
- overall_pass: true only if average score >= 3.5 AND correctness >= 3

Return ONLY valid JSON (no markdown):
{{
  "dimensions": {{
    "correctness": {{"score": 1, "reason": "..."}},
    "groundedness": {{"score": 1, "reason": "..."}},
    "completeness": {{"score": 1, "reason": "..."}},
    "helpfulness": {{"score": 1, "reason": "..."}},
    "tone": {{"score": 1, "reason": "..."}}
  }},
  "overall_pass": false,
  "summary": "one sentence overall assessment"
}}"""

        system = (
            "You are a strict but fair LLM judge for retail QA evaluation. "
            "Output valid JSON only."
        )
        raw = self.llm.generate(prompt, system_prompt=system)

        try:
            parsed = _extract_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            return {
                "dimensions": {
                    name: {"score": 1, "reason": f"Judge parse error: {e}"}
                    for name in self.dimension_names
                },
                "overall_pass": False,
                "summary": "Judge failed to return valid JSON.",
                "raw_judge_output": raw,
            }

        dimensions = parsed.get("dimensions") or {}
        normalized: Dict[str, Dict[str, Any]] = {}
        scores: List[float] = []
        for name in self.dimension_names:
            entry = dimensions.get(name) or {}
            score = int(entry.get("score", 1))
            score = max(1, min(5, score))
            normalized[name] = {
                "score": score,
                "reason": str(entry.get("reason", "")),
            }
            scores.append(score)

        avg = sum(scores) / len(scores) if scores else 1.0
        overall_pass = bool(parsed.get("overall_pass"))
        if "overall_pass" not in parsed:
            overall_pass = avg >= 3.5 and normalized.get("correctness", {}).get("score", 1) >= 3

        return {
            "dimensions": normalized,
            "overall_pass": overall_pass,
            "summary": str(parsed.get("summary", "")),
            "average_score": round(avg, 2),
            "raw_judge_output": raw,
        }


_judge_cache: Dict[str, LLMJudge] = {}


def get_judge(provider: Optional[str] = None, model: Optional[str] = None) -> LLMJudge:
    from backend.app.config import config

    key = f"{provider or config.JUDGE_PROVIDER}:{model or config.JUDGE_MODEL or 'default'}"
    if key not in _judge_cache:
        _judge_cache[key] = LLMJudge(provider=provider, model=model)
    return _judge_cache[key]
