"""What the model sees after a tool call (PLAN §8.6).

P9: tool output is untrusted data. It always reaches the model wrapped in a JSON
envelope (``{"ok": true, "data": ...}`` or ``{"ok": false, "error": ...}``) and
capped in size, never spliced into the prompt as free text.
"""

from __future__ import annotations

import json
from typing import Any

from harness.domain.models import ToolErrorInfo, ToolResult


def result_payload(result: ToolResult) -> dict[str, Any]:
    if result.ok:
        return {"ok": True, "data": result.data}
    assert result.error is not None  # guaranteed by ToolResult's validator
    return error_payload(result.error)


def error_payload(error: ToolErrorInfo) -> dict[str, Any]:
    # P7: errors are context. A compact, structured error lets the model adapt.
    return {"ok": False, "error": error.model_dump()}


def render(payload: dict[str, Any], max_chars: int) -> str:
    """Serialise, truncating to ``max_chars`` with a visible marker."""
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}[truncated {len(text) - max_chars} chars]"
