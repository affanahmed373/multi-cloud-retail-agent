"""
LLM provider registry for API and Gradio UI.

Central list of supported providers, display labels, and config checks.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

ProviderChoice = Tuple[str, str]  # (label, provider_id)


PROVIDER_SPECS: List[Dict[str, Any]] = [
    {
        "id": "mock",
        "label": "Mock (local rules)",
        "description": "Rule-based replies, no API key",
        "always_available": True,
    },
    {
        "id": "openai",
        "label": "OpenAI",
        "description": "GPT via OPENAI_API_KEY",
        "env_any": ["OPENAI_API_KEY"],
    },
    {
        "id": "deepseek",
        "label": "DeepSeek",
        "description": "DeepSeek Chat via DEEPSEEK_API_KEY",
        "env_any": ["DEEPSEEK_API_KEY"],
    },
    {
        "id": "gemini",
        "label": "Gemini (Google AI)",
        "description": "Gemini API via GEMINI_API_KEY or GOOGLE_API_KEY",
        "env_any": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    },
    {
        "id": "vertex",
        "label": "Vertex AI (GCP Gemini)",
        "description": "Gemini on Vertex via VERTEX_PROJECT_ID + GCP credentials",
        "env_any": ["VERTEX_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"],
    },
    {
        "id": "bedrock",
        "label": "AWS Bedrock",
        "description": "Claude/Nova on Bedrock via AWS credentials",
        "env_any": ["AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_REGION"],
    },
]

# UI/API aliases
PROVIDER_ALIASES = {
    "google": "gemini",
    "google-ai": "gemini",
    "gcp": "vertex",
    "aws": "bedrock",
}


def normalize_provider(provider: Optional[str]) -> str:
    key = (provider or "mock").strip().lower()
    return PROVIDER_ALIASES.get(key, key)


def is_provider_configured(provider_id: str) -> bool:
    """Best-effort check whether credentials for a provider appear configured."""
    provider_id = normalize_provider(provider_id)
    spec = next((p for p in PROVIDER_SPECS if p["id"] == provider_id), None)
    if spec is None:
        return False
    if spec.get("always_available"):
        return True
    env_any = spec.get("env_any") or []
    return any(os.getenv(name) for name in env_any)


def provider_choices(include_mock: bool = True) -> List[ProviderChoice]:
    """Gradio dropdown choices: (label, provider_id)."""
    choices: List[ProviderChoice] = []
    for spec in PROVIDER_SPECS:
        if not include_mock and spec["id"] == "mock":
            continue
        label = spec["label"]
        if not is_provider_configured(spec["id"]):
            label = f"{label} (needs setup)"
        choices.append((label, spec["id"]))
    return choices


def default_provider_choice(fallback: str = "mock") -> str:
    from .config import config

    preferred = normalize_provider(config.LLM_PROVIDER)
    if preferred in {spec["id"] for spec in PROVIDER_SPECS}:
        return preferred
    return normalize_provider(fallback)


def provider_help_text() -> str:
    lines = ["**Providers** — pick any; missing keys show an error when you ask."]
    for spec in PROVIDER_SPECS:
        status = "ready" if is_provider_configured(spec["id"]) else "needs setup"
        lines.append(f"- **{spec['label']}** ({status}): {spec['description']}")
    return "\n".join(lines)
