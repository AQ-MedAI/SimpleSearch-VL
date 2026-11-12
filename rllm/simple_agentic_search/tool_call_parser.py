"""Shared strict parser for model-emitted tool calls."""

from __future__ import annotations

import json
import re
from typing import Any


_TOOL_CALL_JSON_HINT = (
    'Expected exactly: {"name": "text_search", "arguments": {"query": ["..."]}}. '
    'The top-level JSON object must have "name" and "arguments" as sibling keys. '
    'Do not insert an extra object between "name" and "arguments".'
)


def _json_error_hint(text: str) -> str:
    compact = " ".join(str(text).split())
    if re.search(r'"name"\s*:\s*"[^"]+"\s*,\s*\{\s*"arguments"\s*:', compact):
        return _TOOL_CALL_JSON_HINT
    if 'arguments"' in compact and '"arguments"' not in compact:
        return _TOOL_CALL_JSON_HINT
    return _TOOL_CALL_JSON_HINT


class ToolCallParseError(ValueError):
    """Raised when a <tool_call> block is not valid strict JSON schema."""


def parse_tool_call_json(text: str) -> dict[str, Any]:
    """Parse and validate one strict-JSON tool call object.

    Execution and format reward must use the same parser so a tool call cannot
    execute successfully while receiving a structural format penalty.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolCallParseError(
            f"tool_call invalid JSON: {exc.msg}. {_json_error_hint(text)}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ToolCallParseError("tool_call must be a JSON object")

    name = parsed.get("name")
    if not isinstance(name, str) or not name:
        raise ToolCallParseError("tool_call 'name' must be a non-empty string")

    arguments = parsed.get("arguments")
    if not isinstance(arguments, dict):
        raise ToolCallParseError("tool_call 'arguments' must be an object")

    return parsed
