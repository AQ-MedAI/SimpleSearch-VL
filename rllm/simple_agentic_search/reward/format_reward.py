"""
Format reward for Mars Agentic Search.

Checks that all assistant turns follow the required format:
  <thinking>...</thinking>
  <tool_call>{"name": ..., "arguments": ...}</tool_call>   (tool call)
  OR
  <thinking>...</thinking>
  <answer>...</answer>                  (final answer)

Structural format errors receive 0.0. One or more tool-argument contract
violations (such as over-budget batched queries) receive a small configurable
trajectory-level penalty.
"""

import logging
import re
from typing import Any

from simple_agentic_search.tool_call_parser import ToolCallParseError, parse_tool_call_json

logger = logging.getLogger(__name__)

# Must match MAX_LLM_CALL_PER_RUN in deepresearch_agent.py
MAX_ROUNDS = 10
DEFAULT_TOOL_CONTRACT_PENALTY = 0.2


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _get_tool_limits(tools_config: dict | None) -> dict[str, int]:
    cfg = _as_dict(tools_config)
    image_cfg = _as_dict(cfg.get("image_search"))
    zoom_cfg = _as_dict(cfg.get("zoom_in"))
    return {
        "text_search": int(cfg.get("search_max_queries", 3)),
        "visit": int(cfg.get("visit_max_urls", 3)),
        "image_search": int(image_cfg.get("max_bboxes", cfg.get("max_bboxes", 3))),
        "zoom_in": int(zoom_cfg.get("max_bboxes", cfg.get("max_bboxes", 3))),
    }


def _count_items(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if value is None:
        return 0
    return 1


def _count_bboxes(value: Any) -> int:
    if not isinstance(value, list) or not value:
        return 0
    if isinstance(value[0], (int, float)):
        return 1
    return len(value)


def _count_tool_bboxes(args: dict) -> int:
    """Count bbox requests for the current bbox_2d schema and legacy regions form."""
    count = _count_bboxes(args.get("bbox_2d"))
    if count:
        return count

    regions = args.get("regions")
    if isinstance(regions, list):
        return len(regions)
    return 0


def _validate_tool_contract(parsed: dict, turn_idx: int, tools_config: dict | None) -> list[str]:
    """Validate budget-like tool argument contracts without treating them as parse failures."""
    errors = []
    name = parsed.get("name")
    args = parsed.get("arguments")
    if not isinstance(args, dict):
        return errors

    limits = _get_tool_limits(tools_config)
    if name == "text_search":
        count = _count_items(args.get("query"))
        limit = limits["text_search"]
        if count > limit:
            errors.append(
                f"Turn {turn_idx}: text_search query count {count} exceeds limit {limit}"
            )
    elif name == "visit":
        count = _count_items(args.get("url"))
        limit = limits["visit"]
        if count > limit:
            errors.append(
                f"Turn {turn_idx}: visit URL count {count} exceeds limit {limit}"
            )
    elif name in ("image_search", "zoom_in"):
        count = _count_tool_bboxes(args)
        limit = limits[name]
        if count > limit:
            errors.append(
                f"Turn {turn_idx}: {name} bbox count {count} exceeds limit {limit}"
            )
    return errors


def _validate_turn(
    content: str,
    is_final: bool,
    turn_idx: int,
    tools_config: dict | None = None,
) -> tuple[list[str], list[str]]:
    """Validate a single assistant turn.

    Returns (structural_errors, contract_errors).
    """
    errors = []
    contract_errors = []

    # 1. Must have balanced <thinking>...</thinking>
    n_open = content.count("<thinking>")
    n_close = content.count("</thinking>")
    if n_open == 0:
        errors.append(f"Turn {turn_idx}: missing <thinking> tag")
    if n_open != n_close:
        errors.append(f"Turn {turn_idx}: unbalanced <thinking> tags ({n_open} open, {n_close} close)")

    # Extract content after </thinking>
    after_thinking = content.split("</thinking>")[-1].strip() if "</thinking>" in content else content.strip()

    tool_call_count = after_thinking.count("<tool_call>")
    answer_count = after_thinking.count("<answer>")
    has_tool_call = "<tool_call>" in after_thinking and "</tool_call>" in after_thinking
    has_answer = "<answer>" in after_thinking and "</answer>" in after_thinking

    # 2. Must have exactly one action: tool_call XOR answer
    if not has_tool_call and not has_answer:
        errors.append(f"Turn {turn_idx}: missing both <tool_call> and <answer>")
    elif has_tool_call and has_answer:
        errors.append(f"Turn {turn_idx}: has both <tool_call> and <answer> (must be exactly one)")
    if tool_call_count > 1:
        errors.append(f"Turn {turn_idx}: has multiple <tool_call> blocks; use one batched tool call instead")
    if answer_count > 1:
        errors.append(f"Turn {turn_idx}: has multiple <answer> blocks")

    # 3. Validate tool_call JSON
    if has_tool_call:
        tc_match = re.search(r"<tool_call>(.*?)</tool_call>", after_thinking, re.DOTALL)
        if tc_match:
            tc_text = tc_match.group(1).strip()
            try:
                parsed = parse_tool_call_json(tc_text)
                if not errors:
                    contract_errors.extend(
                        _validate_tool_contract(parsed, turn_idx, tools_config)
                    )
            except ToolCallParseError as exc:
                errors.append(f"Turn {turn_idx}: {exc}")

    # 4. Validate answer content
    if has_answer:
        ans_match = re.search(r"<answer>(.*?)</answer>", after_thinking, re.DOTALL)
        if ans_match:
            ans_text = ans_match.group(1).strip()
            if not ans_text:
                errors.append(f"Turn {turn_idx}: empty <answer> content")

    return errors, contract_errors


def compute_format_reward(
    messages: list[dict],
    weight: float = 0.5,
    tools_config: dict | None = None,
    contract_penalty: float = DEFAULT_TOOL_CONTRACT_PENALTY,
) -> dict:
    """
    Compute format reward from the full message trajectory.

    Args:
        messages: The full message list from agent result (with role/content).
        weight: Reward value when all format checks pass.
        tools_config: Tool limits from YAML, e.g. visit_max_urls/search_max_queries.
        contract_penalty: Trajectory-level penalty when one or more valid tool
            calls exceed configured argument budgets.

    Returns:
        dict:
            reward: float
            format_errors: list[str]
    """
    assistant_turns = [
        m.get("content", "") for m in messages if m.get("role") == "assistant"
    ]

    structural_errors = []
    contract_errors = []
    for i, turn_content in enumerate(assistant_turns):
        is_final = (i == len(assistant_turns) - 1)
        turn_errors, turn_contract_errors = _validate_turn(
            turn_content, is_final, turn_idx=i + 1, tools_config=tools_config
        )
        structural_errors.extend(turn_errors)
        contract_errors.extend(turn_contract_errors)

    # Round-limit compliance: if model used all available rounds,
    # the last assistant turn must end with <answer>.
    n_turns = len(assistant_turns)
    if n_turns >= MAX_ROUNDS and assistant_turns:
        last_content = assistant_turns[-1]
        if "<answer>" not in last_content or "</answer>" not in last_content:
            structural_errors.append(
                f"Round-limit non-compliance: used all {MAX_ROUNDS} "
                f"rounds but last turn lacks <answer> tags"
            )

    if structural_errors:
        reward = 0.0
    else:
        penalty = max(0.0, float(contract_penalty)) if contract_errors else 0.0
        reward = max(0.0, float(weight) - penalty)

    all_errors = structural_errors + contract_errors
    if all_errors:
        logger.debug("Format errors: %s", all_errors)

    return {
        "reward": reward,
        "format_errors": all_errors,
        "format_structural_errors": structural_errors,
        "tool_contract_errors": contract_errors,
    }
