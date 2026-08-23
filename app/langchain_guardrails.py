"""
LangChain / LangGraph guardrails for the retail agent.

Uses LangChain AgentMiddleware:
- PIIMiddleware for email, credit card, IP, URL (input redact/mask, output redact)
- RetailScopeMiddleware for prompt injection, unsafe, and off-topic requests

Wired as LangGraph nodes in app/graph.py (input_guardrails -> agent -> output_guardrails).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.agents.middleware.pii import PIIDetectionError, PIIMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from .guardrails import (
    REFUSAL_INJECTION,
    REFUSAL_OFF_TOPIC,
    REFUSAL_UNSAFE,
    _has_retail_signal,
    _matches_any,
    _INJECTION_PATTERNS,
    _OFF_TOPIC_PATTERNS,
    _UNSAFE_PATTERNS,
    scoped_system_prompt,
)

PIIStrategy = Literal["block", "redact", "mask", "hash"]
_RUNTIME = Runtime()


@dataclass
class GuardrailOutcome:
    allowed: bool
    text: str
    code: Optional[str] = None
    message: Optional[str] = None
    stage: Optional[str] = None
    pii_detected: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "blocked": not self.allowed,
            "stage": self.stage,
            "code": self.code,
            "pii_detected": self.pii_detected,
        }


class RetailScopeMiddleware(AgentMiddleware):
    """LangChain before_agent guardrail: scope, injection, and unsafe content."""

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, HumanMessage) or not last.content:
            return None

        text = str(last.content).strip()
        if not text:
            return {
                "messages": [
                    AIMessage(
                        content="Please enter a question about our store or products."
                    )
                ],
                "jump_to": "end",
            }

        if _matches_any(text, _INJECTION_PATTERNS):
            return {
                "messages": [AIMessage(content=REFUSAL_INJECTION)],
                "jump_to": "end",
            }

        if _matches_any(text, _UNSAFE_PATTERNS):
            return {
                "messages": [AIMessage(content=REFUSAL_UNSAFE)],
                "jump_to": "end",
            }

        if _matches_any(text, _OFF_TOPIC_PATTERNS) and not _has_retail_signal(text):
            return {
                "messages": [AIMessage(content=REFUSAL_OFF_TOPIC)],
                "jump_to": "end",
            }

        return None


class ApiKeyBlockMiddleware(AgentMiddleware):
    """Block API keys and secrets in user input (custom PIIMiddleware-style detector)."""

    _DETECTOR = r"sk-[a-zA-Z0-9]{20,}|LANGFUSE_(SECRET|PUBLIC)_KEY|aws_secret_access_key"

    def __init__(self) -> None:
        self._pii = PIIMiddleware(
            "api_key",
            detector=self._DETECTOR,
            strategy="block",
            apply_to_input=True,
            apply_to_output=True,
        )

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        try:
            return self._pii.before_model(state, runtime)
        except PIIDetectionError:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "Please do not share API keys or secrets in chat. "
                            "Ask about our products, stock, or store policies instead."
                        )
                    )
                ],
                "jump_to": "end",
            }

    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        try:
            return self._pii.after_model(state, runtime)
        except PIIDetectionError:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "I couldn't produce a safe answer. "
                            "Please rephrase your question about our store."
                        )
                    )
                ],
            }


class GuardrailPipeline:
    """
    Runs LangChain middleware as input/output guardrail stages for LangGraph.
    """

    def __init__(
        self,
        input_pii_strategy: PIIStrategy = "redact",
        output_pii_strategy: PIIStrategy = "redact",
        credit_card_strategy: PIIStrategy = "mask",
    ) -> None:
        self.scope = RetailScopeMiddleware()
        self.api_key = ApiKeyBlockMiddleware()
        self.input_pii = [
            PIIMiddleware(
                "email",
                strategy=input_pii_strategy,
                apply_to_input=True,
            ),
            PIIMiddleware(
                "credit_card",
                strategy=credit_card_strategy,
                apply_to_input=True,
            ),
            PIIMiddleware("ip", strategy=input_pii_strategy, apply_to_input=True),
            PIIMiddleware("url", strategy=input_pii_strategy, apply_to_input=True),
        ]
        self.output_pii = [
            PIIMiddleware(
                "email",
                strategy=output_pii_strategy,
                apply_to_output=True,
            ),
            PIIMiddleware(
                "credit_card",
                strategy=credit_card_strategy,
                apply_to_output=True,
            ),
            PIIMiddleware("ip", strategy=output_pii_strategy, apply_to_output=True),
            PIIMiddleware("url", strategy=output_pii_strategy, apply_to_output=True),
        ]

    def _pii_types_from_update(self, update: dict[str, Any]) -> List[str]:
        detected: List[str] = []
        for msg in update.get("messages") or []:
            content = str(getattr(msg, "content", "") or "")
            for label in ("EMAIL", "CREDIT_CARD", "IP", "URL", "API_KEY"):
                if f"[REDACTED_{label}]" in content or label.lower() in content.lower():
                    detected.append(label.lower())
        return list(dict.fromkeys(detected))

    def process_input(self, query: str) -> GuardrailOutcome:
        """Input stage: scope checks + PII redaction before the agent runs."""
        text = str(query or "").strip()
        state: AgentState = {"messages": [HumanMessage(content=text)]}

        scope_update = self.scope.before_agent(state, _RUNTIME)
        if scope_update and scope_update.get("jump_to") == "end":
            refusal = str(scope_update["messages"][-1].content)
            code = "injection" if REFUSAL_INJECTION in refusal else "scope"
            if REFUSAL_UNSAFE in refusal:
                code = "unsafe"
            elif REFUSAL_OFF_TOPIC in refusal:
                code = "off_topic"
            elif "Please enter" in refusal:
                code = "empty"
            return GuardrailOutcome(
                allowed=False,
                text=text,
                code=code,
                message=refusal,
                stage="input",
            )

        api_update = self.api_key.before_agent(state, _RUNTIME)
        if api_update and api_update.get("jump_to") == "end":
            return GuardrailOutcome(
                allowed=False,
                text=text,
                code="api_key",
                message=str(api_update["messages"][-1].content),
                stage="input",
            )

        pii_detected: List[str] = []
        current = text
        for middleware in self.input_pii:
            msg_state: AgentState = {"messages": [HumanMessage(content=current)]}
            try:
                update = middleware.before_model(msg_state, _RUNTIME)
            except PIIDetectionError:
                return GuardrailOutcome(
                    allowed=False,
                    text=current,
                    code="pii",
                    message=(
                        "Your message contains sensitive personal data we can't process. "
                        "Please remove it and ask again without sharing private details."
                    ),
                    stage="input",
                )
            if update:
                pii_detected.extend(self._pii_types_from_update(update))
                current = str(update["messages"][-1].content)

        return GuardrailOutcome(
            allowed=True,
            text=current,
            stage="input",
            pii_detected=pii_detected,
        )

    def process_output(self, answer: str) -> GuardrailOutcome:
        """Output stage: PII redaction and secret blocking before returning."""
        text = str(answer or "").strip()
        if not text:
            return GuardrailOutcome(
                allowed=False,
                text=text,
                code="empty_output",
                message=(
                    "I couldn't produce a safe answer. "
                    "Please rephrase your question about our store."
                ),
                stage="output",
            )

        state: AgentState = {"messages": [AIMessage(content=text)]}
        pii_detected: List[str] = []
        current = text

        api_update = self.api_key.after_model(state, _RUNTIME)
        if api_update:
            if any(
                "couldn't produce a safe answer" in str(m.content)
                for m in api_update.get("messages", [])
            ):
                return GuardrailOutcome(
                    allowed=False,
                    text=current,
                    code="api_key",
                    message=str(api_update["messages"][-1].content),
                    stage="output",
                )
            pii_detected.extend(self._pii_types_from_update(api_update))
            current = str(api_update["messages"][-1].content)

        for middleware in self.output_pii:
            msg_state: AgentState = {"messages": [AIMessage(content=current)]}
            try:
                update = middleware.after_model(msg_state, _RUNTIME)
            except PIIDetectionError:
                return GuardrailOutcome(
                    allowed=False,
                    text=current,
                    code="pii",
                    message=(
                        "I couldn't produce a safe answer. "
                        "Please rephrase your question about our store."
                    ),
                    stage="output",
                )
            if update:
                pii_detected.extend(self._pii_types_from_update(update))
                current = str(update["messages"][-1].content)

        return GuardrailOutcome(
            allowed=True,
            text=current,
            stage="output",
            pii_detected=pii_detected,
        )


_default_pipeline: Optional[GuardrailPipeline] = None


def get_guardrail_pipeline() -> GuardrailPipeline:
    global _default_pipeline
    if _default_pipeline is None:
        from .config import config

        _default_pipeline = GuardrailPipeline(
            input_pii_strategy=config.PII_INPUT_STRATEGY,
            output_pii_strategy=config.PII_OUTPUT_STRATEGY,
            credit_card_strategy=config.PII_CREDIT_CARD_STRATEGY,
        )
    return _default_pipeline


__all__ = [
    "GuardrailOutcome",
    "GuardrailPipeline",
    "RetailScopeMiddleware",
    "get_guardrail_pipeline",
    "scoped_system_prompt",
]
