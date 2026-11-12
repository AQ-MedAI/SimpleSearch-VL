"""
Multi-turn ReAct agent for Mars Agentic Search.

Key features:
- <thinking> / <tool_call> / <answer> format with Qwen3-VL native tool schema
- Force final answer on repetition/round limit
- Tool call retry with backoff
- zoom_in tool support (image injection into conversation)
- Override rollout engine prompt length to allow multi-turn context accumulation
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import re
import time
import uuid
from collections import Counter
from datetime import datetime
from typing import Optional


from PIL import Image

from rllm.engine.rollout import RolloutEngine

from simple_agentic_search.prompts import (
    SYSTEM_PROMPT_TEXT,
    SYSTEM_PROMPT_VISION,
    build_system_prompt,
    TOOL_RESPONSE_GUIDANCE,
    SEARCH_RESULT_GUIDANCE,
    IMAGE_SEARCH_RESULT_GUIDANCE,
    IMAGE_SEARCH_RESULT_GUIDANCE_WITH_THUMBNAILS,
    FORMAT_ERROR_MSG,
    REPETITION_ERROR_MSG,
    FINAL_ANSWER_INSTRUCTION,
    ROUND_LIMIT_INSTRUCTION_TEMPLATE,
    AGENT_TIMEOUT_FINAL_ANSWER_INSTRUCTION,
)
from simple_agentic_search.tools.shared import ToolCallResult
from simple_agentic_search.tool_call_parser import ToolCallParseError, parse_tool_call_json

logger = logging.getLogger(__name__)
_CTX_AGENT_REQUEST_ID_BASE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_request_id_base", default=None
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_LLM_CALL_PER_RUN = 10
MAX_REPEAT_TURN = 3
# Per-run context limits
MAX_PROMPT_LENGTH_PER_RUN = 32000
MAX_RESPONSE_LENGTH_PER_RUN = 4096
# Timeout for a single tool call (seconds)
TOOL_CALL_TIMEOUT = 120
# Timeout for the entire agent run (seconds)
AGENT_RUN_TIMEOUT = 360


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def analyze_repetition_ngram(
    text: str, n: int = 30, threshold: float = 0.5
) -> bool:
    if not text or len(text) < n:
        return False
    ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
    total_count = len(ngrams)
    if total_count == 0:
        return False
    unique_count = len(Counter(ngrams))
    return (unique_count / total_count) < threshold


def count_words(text: str) -> int:
    pattern = re.compile(r"[A-Za-z]+(?:['-][A-Za-z]+)*")
    return len(pattern.findall(text))


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def build_input_image_refs(images: list | None, image_path=None) -> dict[str, str]:
    """Assign stable img_N IDs to input images and keep their raw path when known."""
    image_items = list(images or [])
    raw_paths = [str(path) for path in _as_list(image_path) if path]
    refs: dict[str, str] = {}
    for idx, image in enumerate(image_items):
        source = raw_paths[idx] if idx < len(raw_paths) else ""
        if not source and isinstance(image, str):
            source = image
        if not source and idx == 0 and raw_paths:
            source = raw_paths[0]
        refs[f"img_{idx}"] = source
    return refs


def _strip_legacy_image_prefix(question: str) -> str:
    return re.sub(
        r"^\s*(?:<image>\s*)?image_id:\s*[^\n]+\n?\s*Question:\s*",
        "",
        str(question),
        count=1,
        flags=re.IGNORECASE,
    )


def add_image_id_context(question: str, image_refs: dict[str, str]) -> str:
    if not image_refs:
        return question
    clean_question = _strip_legacy_image_prefix(question)
    image_lines = [f"<image> img_idx={idx}" for idx, _ in enumerate(image_refs)]
    return "\n".join(image_lines) + "\n\n" + f"Question: {clean_question}"



def _resolve_image_search_image_id(raw_image_id, image_refs: dict[str, str]) -> tuple[str, str]:
    if not image_refs:
        image_id = "" if raw_image_id in (None, "") else str(raw_image_id)
        return image_id, image_id
    if raw_image_id in (None, ""):
        display_id = next(iter(image_refs))
        return image_refs[display_id], display_id
    display_id = str(raw_image_id)
    if display_id in image_refs:
        return image_refs[display_id], display_id
    if display_id.startswith("img_"):
        known = ", ".join(image_refs.keys())
        raise ValueError(f"unknown image_id(s): {[display_id]}; available image IDs: {known}")
    # Backward compatibility: allow raw image paths.
    return display_id, display_id


def _normalize_region_bbox(raw_bbox):
    if raw_bbox is None:
        return None
    if isinstance(raw_bbox, list) and raw_bbox and isinstance(raw_bbox[0], (int, float)):
        return raw_bbox
    return raw_bbox


def _image_id_from_img_idx(raw_img_idx, image_refs: dict[str, str]) -> str:
    if raw_img_idx in (None, ""):
        return ""
    try:
        idx = int(raw_img_idx)
    except (TypeError, ValueError):
        raise ValueError(f"img_idx must be a number, got {raw_img_idx!r}")
    display_id = f"img_{idx}"
    if display_id not in image_refs:
        known = ", ".join(str(i) for i in range(len(image_refs)))
        raise ValueError(f"unknown img_idx: {idx}; available img_idx values: {known}")
    return display_id


def resolve_image_search_regions(tool_args: dict, image_refs: dict[str, str]) -> list[dict]:
    """Resolve model-facing img_N IDs into internal paths while preserving display IDs."""
    raw_regions = tool_args.get("regions")
    if raw_regions is not None:
        if not isinstance(raw_regions, list) or not raw_regions:
            raise ValueError("regions must be a non-empty list")
        regions = []
        for idx, region in enumerate(raw_regions, 1):
            if not isinstance(region, dict):
                raise ValueError(f"regions[{idx}] must be an object")
            raw_image_id = region.get("image_id")
            if raw_image_id in (None, "") and "img_idx" in region:
                raw_image_id = _image_id_from_img_idx(region.get("img_idx"), image_refs)
            image_path, display_id = _resolve_image_search_image_id(raw_image_id, image_refs)
            bbox = _normalize_region_bbox(region.get("bbox_2d"))
            if bbox is None:
                raise ValueError(f"regions[{idx}].bbox_2d is required")
            regions.append({"image_id": image_path, "display_image_id": display_id, "bbox_2d": bbox})
        return regions

    raw_image_ids = tool_args.get("image_id")
    raw_bbox = tool_args.get("bbox_2d")
    image_values = _as_list(raw_image_ids) if raw_image_ids not in (None, "") else [None]
    if raw_bbox is None:
        raise ValueError("regions is required; legacy bbox_2d was missing")
    if isinstance(raw_bbox, list) and raw_bbox and isinstance(raw_bbox[0], (int, float)):
        bbox_values = [raw_bbox]
    else:
        bbox_values = raw_bbox
    if not isinstance(bbox_values, list) or not bbox_values:
        raise ValueError("bbox_2d must be a non-empty bbox or bbox list")

    if len(image_values) == 1:
        pairs = [(image_values[0], bbox) for bbox in bbox_values]
    elif len(bbox_values) == 1:
        pairs = [(image_value, bbox_values[0]) for image_value in image_values]
    elif len(image_values) == len(bbox_values):
        pairs = list(zip(image_values, bbox_values))
    else:
        raise ValueError(
            "legacy image_id and bbox_2d are ambiguous; use regions=[{image_id, bbox_2d}, ...]"
        )

    regions = []
    for image_value, bbox in pairs:
        image_path, display_id = _resolve_image_search_image_id(image_value, image_refs)
        regions.append({"image_id": image_path, "display_image_id": display_id, "bbox_2d": bbox})
    return regions


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class MultiTurnReactAgent:
    """Multi-turn ReAct Agent with <thinking>/<tool_call>/<answer> format."""

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        tools: dict = None,
        system_prompt: str | None = None,
        default_max_tries: int = 3,
        agent_config: dict | None = None,
        enabled_tools: list[str] | None = None,
        **kwargs,
    ):
        self.rollout_engine = rollout_engine
        self.tools = tools or {}
        self.system_prompt = system_prompt
        self.enabled_tools = enabled_tools
        self.tools_config = kwargs.get("tools_config") or {}

        # Agent config: use provided values or fall back to module-level defaults
        cfg = agent_config or {}
        self.image_search_first = bool(cfg.get("image_search_first", False))
        self.default_max_tries = cfg.get("tool_max_retries", default_max_tries)
        self.llm_max_retries = int(cfg.get("llm_max_retries", self.default_max_tries))
        self.llm_call_timeout = cfg.get("llm_call_timeout", None)
        if self.llm_call_timeout is not None:
            self.llm_call_timeout = float(self.llm_call_timeout)
        self.max_llm_calls = cfg.get("max_llm_calls", MAX_LLM_CALL_PER_RUN)
        self.max_repeat_turn = cfg.get("max_repeat_turn", MAX_REPEAT_TURN)
        self.max_prompt_tokens = cfg.get("max_prompt_tokens", MAX_PROMPT_LENGTH_PER_RUN)
        self.max_response_tokens = cfg.get("max_response_tokens", MAX_RESPONSE_LENGTH_PER_RUN)
        self.tool_call_timeout = cfg.get("tool_call_timeout", TOOL_CALL_TIMEOUT)
        self.agent_run_timeout = cfg.get("agent_run_timeout", AGENT_RUN_TIMEOUT)
        self.force_final_answer_timeout = cfg.get("force_final_answer_timeout", 60)

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._request_id_base = str(uuid.uuid4())
        self._llm_request_seq = 0

        # Wire self-summarization for tools that support EXTRACT_MODEL=self.
        # Do not pass tool_call_timeout into tools: cancelling a visit/self-summary
        # call mid-flight can leave orphaned SGLang requests. Let tools finish and
        # rely on their own HTTP/model limits.
        for tool in self.tools.values():
            if hasattr(tool, "set_model_caller"):
                tool.set_model_caller(self.rollout_engine)
            if hasattr(tool, "set_model_timeout"):
                tool.set_model_timeout(None)

    def _set_request_context(self, application_id: str | None = None) -> None:
        base = str(application_id or uuid.uuid4())
        self._request_id_base = re.sub(r"[^A-Za-z0-9_.:-]", "_", base)[:160]
        _CTX_AGENT_REQUEST_ID_BASE.set(self._request_id_base)
        self._llm_request_seq = 0

        for tool in self.tools.values():
            if hasattr(tool, "set_model_request_prefix"):
                tool.set_model_request_prefix(self._request_id_base)

    def _next_model_request_id(self, label: str) -> str:
        base = _CTX_AGENT_REQUEST_ID_BASE.get() or self._request_id_base
        return f"{base}:{label}:{uuid.uuid4().hex}"

    def _get_system_prompt(self, has_images: bool) -> str:
        if self.system_prompt:
            return self.system_prompt
        if self.enabled_tools is not None:
            return build_system_prompt(
                enabled_tools=self.enabled_tools,
                has_image=has_images,
                tools_config=self.tools_config,
            )
        return build_system_prompt(
            mode="vision" if has_images else "text",
            has_image=has_images,
            tools_config=self.tools_config,
        )

    # ------------------------------------------------------------------
    # Optional scripted first step
    # ------------------------------------------------------------------
    async def _run_image_search_first(self, messages, image_refs: dict[str, str]) -> bool:
        """Prepend a config-driven whole-image search before the model loop."""
        if not self.image_search_first or not image_refs:
            return False
        if "image_search" not in self.tools:
            logger.warning("image_search_first is enabled but image_search is unavailable; skipping.")
            return False

        regions_arg = [
            {"img_idx": idx, "bbox_2d": [0, 0, 1000, 1000]}
            for idx in range(len(image_refs))
        ]
        tool_call = {
            "name": "image_search",
            "arguments": {"regions": regions_arg},
        }
        content = (
            "<thinking>I should first use image search on the full input image(s) "
            "to gather visual evidence.</thinking>\n"
            f"<tool_call>\n{json.dumps(tool_call, ensure_ascii=False)}\n</tool_call>"
        )
        assistant_message = {
            "role": "assistant",
            "content": content,
            "step_error": False,
        }
        messages.append(assistant_message)

        try:
            regions = resolve_image_search_regions({"regions": regions_arg}, image_refs)
            raw_result = await self.custom_call_tool("image_search", {"regions": regions})
            if isinstance(raw_result, ToolCallResult):
                result = raw_result.text
                tool_images = raw_result.images
                tool_image_urls = raw_result.image_urls
            else:
                result = str(raw_result)
                tool_images = []
                tool_image_urls = []
        except Exception as exc:  # noqa: BLE001
            logger.exception("image_search_first failed: %s", exc)
            result = "[Tool Execution Error]: image_search_first failed during execution."
            tool_images = []
            tool_image_urls = []
            assistant_message["step_error"] = True

        guidance = (
            IMAGE_SEARCH_RESULT_GUIDANCE_WITH_THUMBNAILS
            if tool_images
            else IMAGE_SEARCH_RESULT_GUIDANCE
        )
        tool_response_text = f"<tool_response>\n{result}\n</tool_response>\n{guidance}"
        msg = {"role": "user", "content": tool_response_text}
        if tool_images:
            msg["images"] = tool_images
            msg["_embed_images"] = True
        if tool_image_urls:
            msg["thumbnail_urls"] = tool_image_urls
        messages.append(msg)
        return True

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------
    async def _abort_request(self, request_id: str | None, reason: str) -> None:
        if not request_id:
            return
        abort = getattr(self.rollout_engine, "abort_request", None)
        if abort is None:
            logger.warning(
                "Cannot abort rollout request %s after %s: rollout engine has no per-request abort.",
                request_id,
                reason,
            )
            return
        try:
            await abort(request_id)
            logger.warning("Aborted rollout request %s after %s.", request_id, reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to abort rollout request %s after %s: %s", request_id, reason, exc)

    async def call_server(self, messages: list[dict], max_tries: Optional[int] = None, **kwargs):
        if max_tries is None:
            max_tries = self.llm_max_retries

        # Override rollout engine limits to allow multi-turn context
        if hasattr(self.rollout_engine, "max_prompt_length"):
            self.rollout_engine.max_prompt_length = int(self.max_prompt_tokens)
        if hasattr(self.rollout_engine, "max_response_length"):
            self.rollout_engine.max_response_length = int(self.max_response_tokens)

        kwargs.pop("max_new_tokens", None)
        kwargs["max_tokens"] = int(self.max_response_tokens)

        last_exc = None
        timeout = self.llm_call_timeout if self.llm_call_timeout and self.llm_call_timeout > 0 else None
        for attempt in range(1, max_tries + 1):
            request_id = self._next_model_request_id("agent")
            call_kwargs = dict(kwargs)
            call_kwargs["application_id"] = request_id
            try:
                coro = self.rollout_engine.get_model_response(messages=messages, **call_kwargs)
                response = await asyncio.wait_for(coro, timeout=timeout) if timeout else await coro
                if hasattr(response, "prompt_length") and hasattr(response, "completion_length"):
                    self.total_prompt_tokens += response.prompt_length
                    self.total_completion_tokens += response.completion_length
                return response
            except asyncio.TimeoutError as exc:
                last_exc = exc
                logger.warning(
                    "call_server timed out after %.1fs (attempt %d/%d); aborting rollout request %s.",
                    timeout or -1,
                    attempt,
                    max_tries,
                    request_id,
                )
                await self._abort_request(request_id, "LLM call timeout")
                if attempt < max_tries:
                    await asyncio.sleep(min(2**attempt, 30))
            except asyncio.CancelledError:
                await self._abort_request(request_id, "LLM call cancellation")
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.debug("call_server failed (attempt %d/%d): %s", attempt, max_tries, exc)
                if attempt < max_tries:
                    await asyncio.sleep(min(2**attempt, 30))
        raise last_exc

    def record_token_usage(self, response) -> None:
        for attr, target in [("prompt_length", "total_prompt_tokens"),
                             ("completion_length", "total_completion_tokens")]:
            val = getattr(response, attr, None)
            if val is not None:
                try:
                    setattr(self, target, int(val))
                except (TypeError, ValueError):
                    setattr(self, target, 0)

    # ------------------------------------------------------------------
    # Result assembly
    # ------------------------------------------------------------------
    def _build_result(self, *, question, answer, messages, prediction, termination, rounds, start_time) -> dict:
        return {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "rounds": rounds,
            "time_taken": time.time() - start_time,
            "token_usage": {
                "prompt": self.total_prompt_tokens,
                "completion": self.total_completion_tokens,
                "max_prompt": self.max_prompt_tokens,
            },
        }

    # ------------------------------------------------------------------
    # Force final answer
    # ------------------------------------------------------------------
    async def _force_final_answer(
        self,
        messages,
        instruction,
        round_num,
        label,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        messages.append({"role": "user", "content": instruction})
        logger.debug("Round %d: %s, requesting final answer.", round_num, label)
        try:
            coro = self.call_server(messages, max_tries=1)
            response = await asyncio.wait_for(coro, timeout=timeout) if timeout else await coro
            final_content = response.text if hasattr(response, "text") and response.text else ""
            messages.append({"role": "assistant", "content": final_content.strip()})
            if "<answer>" in final_content and "</answer>" in final_content:
                prediction = final_content.split("<answer>")[1].split("</answer>")[0].strip()
            else:
                prediction = final_content.strip() if final_content.strip() else "No answer found."
            return prediction, "answer"
        except asyncio.TimeoutError:
            return "No answer found due to final-answer timeout.", "timeout"
        except Exception as exc:  # noqa: BLE001
            return f"No answer found due to model failure: {exc}", "answer"

    # ------------------------------------------------------------------
    # Main reasoning loop
    # ------------------------------------------------------------------
    async def _run(
        self,
        question: str,
        answer: str = None,
        images: list = None,
        image_path: str = None,
        stop_event: asyncio.Event = None,
        **kwargs,
    ) -> dict:
        start_time = time.time()
        self._set_request_context(kwargs.pop("application_id", None))

        has_images = bool(images)
        system_prompt = self._get_system_prompt(has_images)

        # Build user message with stable img_N labels for native multi-image tool calls.
        image_refs = build_input_image_refs(images, image_path) if images else {}
        image_objects = {f"img_{idx}": image for idx, image in enumerate(images or [])}
        if images:
            question = add_image_id_context(question, image_refs)
            user_message = {"role": "user", "content": question, "images": images}
        else:
            user_message = {"role": "user", "content": question}

        messages = [
            {"role": "system", "content": system_prompt},
            user_message,
        ]

        num_llm_calls_available = self.max_llm_calls
        round_num = 0
        termination = None
        prediction = ""
        repetition_count = 0
        self._stop_event = stop_event

        await self._run_image_search_first(messages, image_refs)

        while num_llm_calls_available > 0:
            # Cooperative stop: check if dynamic rollout has signalled early exit
            if stop_event is not None and stop_event.is_set():
                logger.debug("Round %d: stop_event set, exiting early.", round_num)
                return self._build_result(
                    question=question, answer=answer, messages=messages,
                    prediction=prediction or "Stopped early by dynamic rollout.",
                    termination="stopped_early",
                    rounds=round_num, start_time=start_time,
                )

            round_num += 1
            num_llm_calls_available -= 1

            # Agent-level timeout: stop using tools/reasoning and ask for one final
            # answer from the context gathered so far. The outer workflow timeout
            # remains the hard kill switch via asyncio.wait_for in the workflow.
            if time.time() - start_time > self.agent_run_timeout:
                logger.warning(
                    "Round %d: Agent timeout (%ds), forcing final answer.",
                    round_num, self.agent_run_timeout,
                )
                prediction, termination = await self._force_final_answer(
                    messages,
                    AGENT_TIMEOUT_FINAL_ANSWER_INSTRUCTION,
                    round_num,
                    "Agent runtime budget reached",
                    timeout=float(self.force_final_answer_timeout),
                )
                return self._build_result(
                    question=question, answer=answer, messages=messages,
                    prediction=prediction, termination=termination,
                    rounds=round_num, start_time=start_time,
                )

            try:
                response = await self.call_server(messages, **kwargs)
            except Exception:  # noqa: BLE001
                prediction = "call_server failed"
                termination = "error"
                return self._build_result(
                    question=question, answer=answer, messages=messages,
                    prediction=prediction, termination=termination,
                    rounds=round_num, start_time=start_time,
                )

            self.record_token_usage(response)
            content = response.text if hasattr(response, "text") and response.text else ""
            
            # Strip leaked tool_response
            if "<tool_response>" in content:
                content = content[: content.find("<tool_response>")]

            # ---- Branch 1: Tool call ----
            if "<tool_call>" in content and "</tool_call>" in content:
                assistant_message = {
                    "role": "assistant",
                    "content": content.strip(),
                    "step_error": False,
                }
                messages.append(assistant_message)

                tool_call_text = content.split("<tool_call>")[1].split("</tool_call>")[0]
                tool_error = False
                result = ""

                raw_result = ""
                tool_images = []
                tool_image_urls = []
                prompt_tool_name = ""

                try:
                    tool_call = parse_tool_call_json(tool_call_text)
                    prompt_tool_name = tool_call["name"]
                    tool_name = prompt_tool_name
                    tool_args = dict(tool_call["arguments"])

                    # Prompt name -> backend tool name mapping
                    if tool_name == "image_search":
                        try:
                            regions = resolve_image_search_regions(tool_args, image_refs)
                        except ValueError as exc:
                            result = f"[Tool Format Error]: {exc}"
                            tool_error = True
                            raw_result = result
                        else:
                            tool_args = {"regions": regions}
                            if "goal" in tool_call["arguments"]:
                                tool_args["goal"] = tool_call["arguments"].get("goal", "")
                            raw_result = await self.custom_call_tool(tool_name, tool_args)
                    elif tool_name == "zoom_in" and images:
                        try:
                            regions = resolve_image_search_regions(tool_args, image_refs)
                        except ValueError as exc:
                            result = f"[Tool Format Error]: {exc}"
                            tool_error = True
                            raw_result = result
                        else:
                            # Pass source images as parameters; never store them on the tool instance.
                            tool_args = {
                                "regions": regions,
                                "source_images": image_objects,
                                "source_image": images[0],
                            }
                            raw_result = await self.custom_call_tool(tool_name, tool_args)
                    else:
                        raw_result = await self.custom_call_tool(tool_name, tool_args)

                    if tool_error:
                        result = str(raw_result)
                    elif isinstance(raw_result, ToolCallResult):
                        result = raw_result.text
                        tool_images = raw_result.images
                        tool_image_urls = raw_result.image_urls
                    else:
                        result = str(raw_result)

                    # Handle image_search: inject thumbnail images into conversation
                    if tool_name == "image_search" and tool_images:
                        tool_response_text = f"<tool_response>\n{result}\n</tool_response>"
                        guidance = IMAGE_SEARCH_RESULT_GUIDANCE_WITH_THUMBNAILS
                        tool_response_text += f"\n{guidance}"
                        msg = {
                            "role": "user",
                            "content": tool_response_text,
                            "images": tool_images,
                            "_embed_images": True,
                        }
                        if tool_image_urls:
                            msg["thumbnail_urls"] = tool_image_urls
                        messages.append(msg)
                        if tool_error:
                            assistant_message["step_error"] = True
                        continue

                    # Handle zoom_in: inject cropped image into conversation
                    if tool_name == "zoom_in" and tool_images:
                        tool_response_text = f"<tool_response>\n{result}\n</tool_response>"
                        if TOOL_RESPONSE_GUIDANCE:
                            tool_response_text += f"\n{TOOL_RESPONSE_GUIDANCE}"
                        messages.append({
                            "role": "user",
                            "content": tool_response_text,
                            "images": tool_images,
                        })
                        if tool_error:
                            assistant_message["step_error"] = True
                        continue

                except ToolCallParseError as exc:
                    result = f"[Tool Format Error]: {exc}"
                    tool_error = True
                except Exception as exc:
                    logger.exception("Tool call failed: %s", exc)
                    result = "[Tool Execution Error]: Tool call failed during execution."
                    tool_error = True

                if tool_error:
                    assistant_message["step_error"] = True

                tool_response_text = f"<tool_response>\n{result}\n</tool_response>"
                if prompt_tool_name == "image_search":
                    guidance = IMAGE_SEARCH_RESULT_GUIDANCE_WITH_THUMBNAILS if tool_images else IMAGE_SEARCH_RESULT_GUIDANCE
                    tool_response_text += f"\n{guidance}"
                elif TOOL_RESPONSE_GUIDANCE:
                    tool_response_text += f"\n{TOOL_RESPONSE_GUIDANCE}"
                # After text_search, remind model to use visit for detailed page content.
                if prompt_tool_name == "text_search" and SEARCH_RESULT_GUIDANCE:
                    tool_response_text += f"\n{SEARCH_RESULT_GUIDANCE}"
                messages.append({"role": "user", "content": tool_response_text})

            # ---- Branch 2: Final answer ----
            elif "<answer>" in content and "</answer>" in content:
                messages.append({
                    "role": "assistant",
                    "content": content.strip(),
                    "step_error": False,
                })
                prediction = content.split("<answer>")[1].split("</answer>")[0].strip()
                termination = "answer"
                break

            # ---- Branch 3: No tool_call and no answer tags ----
            else:
                is_repetitive = analyze_repetition_ngram(content)
                is_overlong = count_words(content) > 2500
                word_count = count_words(content)

                if is_repetitive and is_overlong:
                    repetition_count += 1
                    logger.debug(
                        "Round %d: repetition detected (count: %d/%d)",
                        round_num, repetition_count, self.max_repeat_turn,
                    )
                    if repetition_count >= self.max_repeat_turn:
                        prediction, termination = await self._force_final_answer(
                            messages, FINAL_ANSWER_INSTRUCTION,
                            round_num, "Content repetition threshold reached",
                        )
                        break

                    messages.append({"role": "assistant", "content": content.strip(), "step_error": True})
                    messages.append({"role": "user", "content": REPETITION_ERROR_MSG})

                elif word_count > 50 and not is_repetitive:
                    # Model produced a substantive response but forgot <answer> tags.
                    messages.append({
                        "role": "assistant",
                        "content": content.strip(),
                        "step_error": True,
                    })
                    prediction = content.strip()
                    termination = "answer"
                    logger.debug(
                        "Round %d: substantive response without <answer> tags, "
                        "auto-extracting as final answer (%d words).",
                        round_num, word_count,
                    )
                    break

                else:
                    messages.append({"role": "assistant", "content": content.strip(), "step_error": True})
                    messages.append({"role": "user", "content": FORMAT_ERROR_MSG})

            # Check cooperative stop before round limit (skip force_final_answer if stopping early)
            if stop_event is not None and stop_event.is_set():
                return self._build_result(
                    question=question, answer=answer, messages=messages,
                    prediction=prediction or "Stopped early by dynamic rollout.",
                    termination="stopped_early",
                    rounds=round_num, start_time=start_time,
                )

            # Check round limit
            if num_llm_calls_available <= 0 and "<answer>" not in content:
                instruction = ROUND_LIMIT_INSTRUCTION_TEMPLATE.format(max_rounds=self.max_llm_calls)
                prediction, termination = await self._force_final_answer(
                    messages, instruction, round_num, "Round limit reached",
                )
                return self._build_result(
                    question=question, answer=answer, messages=messages,
                    prediction=prediction, termination=termination,
                    rounds=round_num, start_time=start_time,
                )

        # Post-loop
        last_msg = messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""
        if not termination:
            if last_msg and "<answer>" in last_msg:
                prediction = last_msg.split("<answer>")[1].split("</answer>")[0]
                termination = "answer"
            else:
                prediction = "No answer found."
                termination = "exceed available llm calls" if num_llm_calls_available == 0 else "answer not found"

        return self._build_result(
            question=question, answer=answer, messages=messages,
            prediction=prediction, termination=termination,
            rounds=round_num, start_time=start_time,
        )

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------
    async def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs) -> "str | ToolCallResult":
        if tool_name not in self.tools:
            err_msg = f"Tool {tool_name} not found. Available tools: {list(self.tools.keys())}"
            logger.error("[ToolError] %s", err_msg)
            return err_msg

        stop_event = getattr(self, "_stop_event", None)
        max_retries = self.default_max_tries
        last_error = None
        for attempt in range(max_retries):
            if stop_event is not None and stop_event.is_set():
                err_msg = f"[Tool Cancelled] {tool_name} skipped (stop_event set, attempt {attempt + 1}/{max_retries})."
                logger.warning("[ToolError] %s | args=%s", err_msg, tool_args)
                return err_msg
            try:
                tool = self.tools[tool_name]
                if hasattr(tool, "call"):
                    if asyncio.iscoroutinefunction(tool.call):
                        coro = tool.call(**tool_args)
                    else:
                        coro = asyncio.get_running_loop().run_in_executor(
                            None, lambda: tool.call(**tool_args)
                        )
                elif callable(tool):
                    coro = asyncio.get_running_loop().run_in_executor(
                        None, lambda: tool(**tool_args)
                    )
                else:
                    err_msg = f"Tool {tool_name} is not callable"
                    logger.error("[ToolError] %s", err_msg)
                    return err_msg
                # No per-call timeout: cancelling tool calls mid-flight (especially
                # self-extract model calls) leaves orphaned requests in SGLang,
                # which crashes the server on release_memory_occupation.
                # External API calls (Jina, search) have their own HTTP timeouts.
                result = await coro
                # Preserve ToolCallResult objects; stringify everything else
                if isinstance(result, ToolCallResult):
                    return result
                return str(result)
            except Exception as e:
                last_error = e
                logger.error(
                    "[ToolError] %s failed (attempt %d/%d): %s | args=%s",
                    tool_name, attempt + 1, max_retries, e, tool_args,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))

        err_msg = f"Error calling tool {tool_name} after {max_retries} retries: {last_error}"
        logger.error("[ToolError] %s", err_msg)
        return err_msg

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def run(self, question: str, answer: str = None, images: list = None,
                  image_path: str = None, **kwargs) -> dict:
        self.reset()
        return await self._run(question, answer, images, image_path, **kwargs)


DeepResearchAgent = MultiTurnReactAgent
