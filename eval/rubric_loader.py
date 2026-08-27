"""Load evaluation rubric from YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_rubric(path: str = "eval/rubric.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_rubric_for_prompt(rubric: Dict[str, Any]) -> str:
    """Render rubric dimensions as judge instructions."""
    lines: List[str] = []
    for dim in rubric.get("dimensions", []):
        name = dim["name"]
        desc = dim.get("description", "")
        lines.append(f"### {name}")
        lines.append(desc)
        scale = dim.get("scale") or {}
        for score in sorted(scale.keys(), key=lambda x: int(x)):
            lines.append(f"  {score}: {scale[score]}")
        lines.append("")
    return "\n".join(lines).strip()
