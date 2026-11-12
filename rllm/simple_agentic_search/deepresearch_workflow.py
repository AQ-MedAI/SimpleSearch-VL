"""
Mars Agentic Search - Training Workflow

Combines the training workflow interface with eval_v2's improvements:
- Configurable prompt styles
- Rollout trajectory saving to disk
- Minimal terminal logging (details in rollout data files)
"""

from io import BytesIO
import asyncio
import json
import logging
import os
import re
import threading
import time
from typing import Any, List, Optional

from PIL import Image

from simple_agentic_search.deepresearch_agent import DeepResearchAgent
from simple_agentic_search.reward.format_reward import compute_format_reward
from simple_agentic_search.reward.accuracy_reward import compute_accuracy_reward
from simple_agentic_search.reward.global_efficiency_reward import GlobalEfficiencyReward
from simple_agentic_search.tools.shared import get_api_usage, reset_api_usage
from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.engine.rollout import RolloutEngine
from rllm.rewards.reward_fn import RewardFunction
from rllm.workflows.workflow import TerminationEvent, TerminationReason, Workflow

import base64

logger = logging.getLogger(__name__)


def as_pil_image(image: Any) -> Image.Image | None:
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, bytes):
        try:
            return Image.open(BytesIO(image)).convert("RGB")
        except Exception:
            return None

    if isinstance(image, str) and image.startswith("image/"):
        try:
            _, encoded = image.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return None

    if isinstance(image, dict):
        if "bytes" in image and image["bytes"] is not None:
            try:
                return Image.open(BytesIO(image["bytes"])).convert("RGB")
            except Exception:
                return None
        data_str = None
        if "data" in image and isinstance(image["data"], str):
            data_str = image["data"]
        elif "path" in image and isinstance(image["path"], str):
            data_str = image["path"]
        elif "url" in image and isinstance(image["url"], str):
            data_str = image["url"]
        if data_str:
            if data_str.startswith("image/"):
                try:
                    _, encoded = data_str.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                    return Image.open(BytesIO(image_bytes)).convert("RGB")
                except Exception:
                    return None
            try:
                return Image.open(data_str).convert("RGB")
            except Exception:
                return None

    if isinstance(image, str):
        try:
            return Image.open(image).convert("RGB")
        except Exception:
            return None

    return None


def _extract_action_from_response(response: str) -> Action:
    if "<tool_call>" in response and "</tool_call>" in response:
        tool_call_text = response.split("<tool_call>")[1].split("</tool_call>")[0]
        return Action(action={"type": "tool_call", "tool_call": tool_call_text.strip()})
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[1].split("</answer>")[0].strip()
        return Action(action={"type": "final_answer", "answer": answer})
    return Action(action={"type": "reasoning", "content": response})


def _extract_tool_payload(tool_call_text: str) -> dict[str, Any]:
    if not isinstance(tool_call_text, str) or not tool_call_text.strip():
        return {}
    try:
        payload = json.loads(tool_call_text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _extract_tool_name(tool_call_text: str) -> str | None:
    payload = _extract_tool_payload(tool_call_text)
    name = payload.get("name") if isinstance(payload, dict) else None
    if name:
        return str(name)
    if not isinstance(tool_call_text, str):
        return None
    match = re.search(r'"name"\s*:\s*"([^"\n]+)"', tool_call_text)
    return match.group(1) if match else None


def _count_list_or_scalar(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if value is None:
        return 0
    return 1


def _count_bbox_requests(args: dict[str, Any]) -> int:
    bbox = args.get("bbox_2d")
    if isinstance(bbox, list) and bbox:
        if isinstance(bbox[0], (int, float)):
            return 1
        return len(bbox)
    regions = args.get("regions")
    if isinstance(regions, list):
        return len(regions)
    return 0


def _expanded_tool_call_count(tool_name: str | None, args: dict[str, Any]) -> int:
    if tool_name == "text_search":
        return max(1, _count_list_or_scalar(args.get("query")))
    if tool_name == "visit":
        return max(1, _count_list_or_scalar(args.get("url")))
    if tool_name in ("image_search", "zoom_in"):
        return max(1, _count_bbox_requests(args))
    return 1


def _extract_tool_behavior_metrics(messages: list[dict]) -> dict[str, Any]:
    tool_sequence = []
    tool_calls_expanded = 0
    expanded_by_tool = {
        "text_search": 0,
        "image_search": 0,
        "visit": 0,
        "zoom_in": 0,
    }
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        action = _extract_action_from_response(msg.get("content", ""))
        action_dict = action.action if isinstance(action.action, dict) else {}
        if action_dict.get("type") != "tool_call":
            continue
        tool_call_text = action_dict.get("tool_call", "")
        payload = _extract_tool_payload(tool_call_text)
        tool_name = _extract_tool_name(tool_call_text)
        if tool_name:
            tool_sequence.append(tool_name)
            args = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
            expanded_count = _expanded_tool_call_count(tool_name, args)
            tool_calls_expanded += expanded_count
            if tool_name in expanded_by_tool:
                expanded_by_tool[tool_name] += expanded_count

    metrics = {
        "tool_sequence_key": ">".join(tool_sequence) if tool_sequence else "none",
        "tool_calls": len(tool_sequence),
        "tool_calls_expanded": tool_calls_expanded,
    }
    for tool_name in ("text_search", "image_search", "visit", "zoom_in"):
        metrics[f"used_{tool_name}"] = 1.0 if tool_name in tool_sequence else 0.0
        metrics[f"{tool_name}_calls"] = tool_sequence.count(tool_name)
        metrics[f"{tool_name}_calls_expanded"] = expanded_by_tool.get(tool_name, 0)
    return metrics


def _is_valid_format(content: str) -> bool:
    if not isinstance(content, str) or not content:
        return False
    pattern = (
        r"^<think>.*?</think>\s*(<tool_call>.*?</tool_call>|<answer>.*?</answer>)\s*$"
    )
    return re.match(pattern, content, re.DOTALL) is not None


def _has_tool_error_observation(observation: Any) -> bool:
    if not isinstance(observation, str):
        return False
    error_markers = (
        "[Json Parse Error]",
    )
    return any(marker in observation for marker in error_markers)


def _is_step_error(step: Step) -> bool:
    if step.info.get("step_error"):
        return True
    return _has_tool_error_observation(step.observation)


def _get_next_observation(messages: list[dict], current_index: int) -> str:
    if current_index + 1 < len(messages):
        next_msg = messages[current_index + 1]
        if next_msg["role"] == "user" and "<tool_response>" in next_msg["content"]:
            return next_msg["content"]
    return ""


def _map_termination_reason(termination: str) -> TerminationReason:
    mapping = {
        "answer": TerminationReason.ENV_DONE,
        "timeout": TerminationReason.UNKNOWN,
        "stopped_early": TerminationReason.UNKNOWN,
        "max_rounds_reached": TerminationReason.UNKNOWN,
        "token_limit_no_answer": TerminationReason.UNKNOWN,
        "answer_token_limit": TerminationReason.UNKNOWN,
        "exceed available llm calls": TerminationReason.UNKNOWN,
        "prompt_budget_reached": TerminationReason.UNKNOWN,
        "max_rounds_reached_no_answer": TerminationReason.UNKNOWN,
        "repetition_detected": TerminationReason.UNKNOWN,
        "tool_errors_too_many": TerminationReason.UNKNOWN,
        "consecutive_step_errors": TerminationReason.UNKNOWN,
        "error": TerminationReason.UNKNOWN,
    }
    return mapping.get(termination, TerminationReason.UNKNOWN)


def _evaluate_answer(prediction: str, ground_truth: str) -> bool:
    if not prediction or not ground_truth:
        return False
    pred = prediction.strip().lower()
    for candidate in ground_truth.split("||"):
        if pred == candidate.strip().lower():
            return True
    return False


def _should_mask_episode(result: dict, episode: Episode) -> tuple[bool, str]:
    steps = episode.trajectories[0].steps if episode.trajectories else []
    termination = result.get("termination", "")

    if termination != "answer":
        return True, termination or "no_final_answer"

    total_steps = len(steps)
    if total_steps > 0:
        step_error_steps = sum(1 for step in steps if _is_step_error(step))
        if step_error_steps / total_steps > 0.5:
            return True, "tool_errors_too_many"

    return False, ""


def _to_pil_image(img: Any) -> Optional[Image.Image]:
    if isinstance(img, Image.Image):
        return img
    pil = as_pil_image(img)
    if pil is not None:
        return pil
    if isinstance(img, dict) and "bytes" in img:
        try:
            return Image.open(BytesIO(img["bytes"])).convert("RGB")
        except Exception:
            return None
    if isinstance(img, str):
        try:
            return Image.open(img).convert("RGB")
        except Exception:
            return None
    return None


def _remove_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _convert_messages_for_json(messages: list[dict]) -> list[dict]:
    """Convert messages to JSON-serializable format (replace PIL images with tags).

    Only embeds base64 image data for messages with _embed_images=True
    (image_search thumbnails). Original images and zoom_in crops keep
    <image> placeholders but no base64 payload to reduce JSON size.
    """
    converted = []
    for msg in messages:
        new_msg = {"role": msg.get("role", ""), "content": ""}
        raw_content = msg.get("content", "")
        images_in_msg = msg.get("images", [])
        if images_in_msg:
            pil_count = sum(1 for img in images_in_msg if isinstance(img, Image.Image))
            existing_count = (raw_content or "").count("<image>")
            if existing_count >= pil_count:
                new_msg["content"] = raw_content or ""
            else:
                placeholders = "<image>" * pil_count
                new_msg["content"] = placeholders + (raw_content or "")
            if msg.get("_embed_images"):
                thumbnail_urls = msg.get("thumbnail_urls", [])
                serialized = _serialize_images(images_in_msg, thumbnail_urls)
                if serialized:
                    new_msg["images"] = serialized
        else:
            new_msg["content"] = raw_content or ""
        converted.append(new_msg)
    return converted


def _serialize_images(raw_images: list | None, thumbnail_urls: list[str] | None = None) -> list:
    """Serialize image list for JSON output (filter out bytes)."""
    if not raw_images:
        return []
    serializable = []
    for i, img in enumerate(raw_images):
        if isinstance(img, dict):
            entry = {k: v for k, v in img.items() if not isinstance(v, bytes)}
            serializable.append(entry)
        elif isinstance(img, Image.Image):
            buf = BytesIO()
            img.save(buf, "JPEG", quality=80)
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            entry = {
                "type": "thumbnail",
                "format": "jpeg",
                "data": b64,
                "size": list(img.size),
            }
            if thumbnail_urls and i < len(thumbnail_urls) and thumbnail_urls[i]:
                entry["url"] = thumbnail_urls[i]
            serializable.append(entry)
        else:
            serializable.append(str(img))
    return serializable


class DeepResearchWorkflow(Workflow):
    # ---- Class-level step tracker (shared across all pooled instances) ----
    _step_lock = threading.Lock()
    _training_step: int = 0  # set by AgentWorkflowEngine.set_training_step

    def __init__(
        self,
        rollout_engine: RolloutEngine,
        executor,
        tools: dict | None = None,
        system_prompt: str | None = None,
        reward_function: RewardFunction | None = None,
        rollout_save_dir: str | None = None,
        rollout_batch_size: int = 128,
        agent_config: dict | None = None,
        enabled_tools: list[str] | None = None,
        reward_config: dict | None = None,
        tools_config: dict | None = None,
        **kwargs,
    ):
        super().__init__(rollout_engine, executor, **kwargs)

        self.tools = tools or {}
        for tool in self.tools.values():
            if hasattr(tool, "set_executor"):
                tool.set_executor(self.executor)
        self.system_prompt = system_prompt
        self.reward_function = reward_function
        self.rollout_save_dir = rollout_save_dir
        self.reward_config = reward_config or {"format": 0.5, "accuracy": 1.0}
        self.tools_config = tools_config or {}
        self.global_efficiency_reward = self._build_global_efficiency_reward()

        self.agent = DeepResearchAgent(
            rollout_engine=rollout_engine,
            tools=self.tools,
            system_prompt=self.system_prompt,
            agent_config=agent_config,
            enabled_tools=enabled_tools,
            tools_config=self.tools_config,
        )

    @classmethod
    def set_step(cls, step: int) -> None:
        """Set the current training step (called by AgentWorkflowEngine)."""
        with cls._step_lock:
            cls._training_step = step

    @staticmethod
    def _parse_uid(uid: str) -> tuple[str, str]:
        """Parse uid 'task_id:rollout_idx' into (group_id, rollout_idx)."""
        if ":" in uid:
            parts = uid.rsplit(":", 1)
            return parts[0], parts[1]
        return uid, "0"

    def _save_rollout_data(
        self,
        uid: str,
        data_id: str,
        result: dict,
        episode: Episode,
        raw_images: list | None,
        answer_nothink: str,
        task: dict,
        timing: dict[str, float] | None = None,
        image_search_match_log: list[dict] | None = None,
        text_search_match_log: list[dict] | None = None,
        visit_match_log: list[dict] | None = None,
        api_usage: dict | None = None,
    ) -> None:
        """Save rollout trajectory data to disk.

        Directory layout:
            rollout_save_dir/{train|val}/step_{N}/sample_{group_id}/rollout_{rollout_idx}.json

        Within each sample directory, all rollouts for the same question are grouped
        together, making it easy to compare different trajectories for one sample.
        """
        if self.rollout_save_dir is None:
            return

        step_num = self._training_step
        mode = "val" if getattr(self.agent.rollout_engine, "validate", False) else "train"
        group_id, rollout_idx = self._parse_uid(uid)

        # Sanitize for filesystem
        safe_group = str(group_id).replace("/", "_").replace(":", "_")
        safe_ridx = str(rollout_idx).replace("/", "_").replace(":", "_")

        sample_dir = os.path.join(
            self.rollout_save_dir, mode, f"step_{step_num}", f"sample_{safe_group}"
        )
        os.makedirs(sample_dir, exist_ok=True)
        save_path = os.path.join(sample_dir, f"rollout_{safe_ridx}.json")

        # Detect which prompt mode was actually used by checking the system message
        messages_list = result.get("messages", [])
        prompt_mode = "unknown"
        if messages_list and messages_list[0].get("role") == "system":
            sys_content = messages_list[0].get("content", "")
            vision_indicators = ("image_search", "zoom_in")
            has_vision_tool = any(ind in sys_content for ind in vision_indicators)
            prompt_mode = "vision" if has_vision_tool else "text"

        record = {
            "uid": uid,
            "group_id": group_id,
            "rollout_idx": rollout_idx,
            "step": step_num,
            "id": data_id,
            "prompt_mode": prompt_mode,
            "messages": _convert_messages_for_json(messages_list),
            "prediction": result.get("prediction", ""),
            "answer_nothink": answer_nothink,
            "answer": task.get("answer", ""),
            "question": task.get("question", ""),
            "data_source": task.get("data_source", ""),
            "data_source_path": task.get("data_source_path", ""),
            "is_correct": episode.is_correct,
            "is_partial": episode.info.get("is_partial", False),
            "termination": result.get("termination", ""),
            "rounds": result.get("rounds", 0),
            "time_taken": result.get("time_taken", 0.0),
            "reward/outcome": episode.metrics.get("reward/outcome", 0.0),
            "reward/accuracy": episode.metrics.get("reward/accuracy", None),
            "reward/format": episode.metrics.get("reward/format", None),
            "reward/global_efficiency": episode.metrics.get("reward/global_efficiency", None),
            "reward/outcome_base_global_efficiency": episode.metrics.get("reward/outcome_base_global_efficiency", None),
            "global_efficiency/reference": episode.metrics.get("global_efficiency/reference", None),
            "global_efficiency/has_reference": episode.metrics.get("global_efficiency/has_reference", None),
            "global_efficiency/ref_rounds": episode.metrics.get("global_efficiency/ref_rounds", None),
            "global_efficiency/ref_tools": episode.metrics.get("global_efficiency/ref_tools", None),
            "global_efficiency/round_bonus": episode.metrics.get("global_efficiency/round_bonus", None),
            "global_efficiency/tool_bonus": episode.metrics.get("global_efficiency/tool_bonus", None),
            "masked": episode.metrics.get("masked", 0.0),
            "mask_reason": episode.info.get("mask_reason", ""),
            "format_errors": episode.info.get("format_errors", []),
            "accuracy_detail": episode.info.get("accuracy_detail", {}),
            "reward_function_reward": episode.info.get("reward_function_reward", None),
            "token_usage": result.get("token_usage", {}),
            "images": _serialize_images(raw_images),
            "timing": timing or {},
            "image_search_match_log": image_search_match_log or [],
            "text_search_match_log": text_search_match_log or [],
            "visit_match_log": visit_match_log or [],
            "api_usage": api_usage or {},
        }

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            logger.debug("Rollout saved: %s", save_path)
        except Exception as exc:
            logger.warning("Failed to save rollout data to %s: %s", save_path, exc)

    def _build_synthetic_error_trajectory(
        self,
        task: dict,
        messages: list[dict] | None,
        reason: str,
        observation: str,
    ) -> Trajectory:
        """Create a one-step failed trajectory so eval keeps the sample in its denominator."""
        safe_messages = list(messages or [])
        if not safe_messages:
            question = task.get("question", task.get("query", "No question provided"))
            if self.system_prompt:
                safe_messages.append({"role": "system", "content": self.system_prompt})
            safe_messages.append({"role": "user", "content": question})

        assistant_content = (
            f"<think>{reason}</think>"
            "<answer>unknown</answer>"
        )
        safe_messages.append({
            "role": "assistant",
            "content": assistant_content,
            "step_error": True,
        })
        step = Step(
            chat_completions=safe_messages,
            model_response=assistant_content,
            action=Action(action={"type": "final_answer", "answer": "unknown"}),
            observation=observation,
            reward=0.0,
        )
        step.info["step_error"] = True
        return Trajectory(
            name="deepresearch_agent",
            task=task,
            steps=[step],
            reward=0.0,
            info={"synthetic_error_step": True, "reason": reason},
        )

    def _build_workflow_timeout_episode(
        self, task: dict, uid: str, elapsed: float
    ) -> Episode:
        """Return a non-empty failed episode for workflow-level timeouts.

        The base Workflow timeout path returns an empty trajectory, which the
        Verl adapter drops with repeat_counts=0. Keeping one synthetic step
        makes validation count the timeout as an incorrect/error item, while
        train still excludes it through compact_filtering.mask_unknown.
        """
        trajectory = self._build_synthetic_error_trajectory(
            task=task,
            messages=None,
            reason="Workflow timed out before completion.",
            observation="[Workflow Timeout]",
        )
        trajectory.info["workflow_timeout"] = True
        messages = trajectory.steps[0].chat_completions

        episode = Episode()
        episode.id = uid
        episode.task = task
        episode.termination_reason = TerminationReason.UNKNOWN
        episode.is_correct = False
        episode.trajectories = [trajectory]
        episode.metrics = {
            "reward/outcome": 0.0,
            "reward/accuracy": 0.0,
            "reward/format": 0.0,
            "rounds": 0,
            "masked": 1.0,
            "completed": 0.0,
            "workflow_timeout": 1.0,
            "error": 1.0,
        }
        episode.info = {
            "mask_reason": "workflow_timeout",
            "time_taken": elapsed,
            "timeout": self.timeout,
            "format_errors": ["workflow_timeout"],
        }

        self._save_rollout_data(
            uid=uid,
            data_id=task.get("id", "") or uid,
            result={
                "messages": messages,
                "prediction": "",
                "termination": "workflow_timeout",
                "rounds": 0,
                "time_taken": elapsed,
                "token_usage": {},
            },
            episode=episode,
            raw_images=task.get("images"),
            answer_nothink=_remove_think_tags(task.get("answer", "")),
            task=task,
            timing={
                "agent_time": round(elapsed, 4),
                "reward_time": 0.0,
                "save_time": 0.0,
                "total_time": round(elapsed, 4),
            },
            api_usage=get_api_usage(),
        )
        return episode

    async def run_with_termination_handling(
        self, task: dict, uid: str, **kwargs
    ) -> Episode:
        self.uid = uid
        self.task = task
        start = time.perf_counter()

        try:
            output = await asyncio.wait_for(
                self.run(task, uid, **kwargs), timeout=self.timeout
            )
            if output is not None and isinstance(output, Episode):
                return output
            return self.postprocess_episode(
                self.collect_trajectories(), TerminationReason.UNKNOWN
            )
        except asyncio.TimeoutError:
            abort = getattr(self.rollout_engine, "abort_all_requests", None)
            if abort is not None:
                try:
                    await abort()
                except Exception:
                    pass
            return self._build_workflow_timeout_episode(
                task, uid, time.perf_counter() - start
            )
        except TerminationEvent as exc:
            return self.postprocess_episode(self.collect_trajectories(), exc.reason)
        except Exception as exc:
            import traceback

            error_details = {
                "error_message": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }
            return self.postprocess_episode(
                self.collect_trajectories(),
                TerminationReason.ERROR,
                error=error_details,
            )


    def _build_global_efficiency_reward(self) -> GlobalEfficiencyReward | None:
        cfg = self.reward_config.get("global_efficiency") if hasattr(self.reward_config, "get") else None
        if not isinstance(cfg, dict) and not hasattr(cfg, "get"):
            return None
        if not bool(cfg.get("enable", False)):
            return None
        state_path = cfg.get("state_path")
        if not state_path:
            logger.warning("global_efficiency is enabled but state_path is empty; reward disabled")
            return None
        return GlobalEfficiencyReward(
            state_path=str(state_path),
            reference=str(cfg.get("reference", "mean")),
            round_weight=float(cfg.get("round_weight", 0.1) or 0.0),
            tool_weight=float(cfg.get("tool_weight", 0.1) or 0.0),
            metric_tool_key=str(cfg.get("metric_tool_key", "tool_calls_expanded")),
        )

    @staticmethod
    def _task_key(task: dict, uid: str) -> str:
        sample_id = task.get("id") or task.get("data_id")
        if sample_id is not None and str(sample_id):
            return str(sample_id)
        return str(uid).rsplit(":", 1)[0]

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task=task, uid=uid)

        question = task.get("question", task.get("query", "No question provided"))
        answer = task.get("answer", "")
        data_id = task.get("id", "") if task.get("id", "") else uid
        answer_nothink = _remove_think_tags(answer)
        total_start = time.perf_counter()
        agent_time = 0.0
        reward_time = 0.0
        save_time = 0.0

        try:
            raw_images = None
            if "images" in task:
                raw_images = task.get("images")

            pil_images: List[Image.Image] = []
            if raw_images is not None:
                if not isinstance(raw_images, list):
                    raw_images = [raw_images]
                for img in raw_images:
                    pil = _to_pil_image(img)
                    if pil is not None:
                        pil_images.append(pil)

            if raw_images and not pil_images:
                img_types = [(type(img).__name__, str(img)[:200]) for img in raw_images]
                logger.warning(
                    "task=%s has %d raw images but PIL conversion failed for all. "
                    "Image types: %s. "
                    "Falling back to text-only prompt (no image tools).",
                    uid, len(raw_images), img_types,
                )

            # Set sample context on image_search tool (for cache-based offline search)
            data_source = task.get("data_source", "")
            sample_idx = task.get("sample_idx", -1)
            # Fallback: sample_idx may be inside extra_info if verl doesn't expose it top-level
            if sample_idx == -1:
                extra_info = task.get("extra_info", {})
                if isinstance(extra_info, dict):
                    sample_idx = extra_info.get("sample_idx", -1)
            if "image_search" in self.tools and hasattr(self.tools["image_search"], "set_sample_context"):
                self.tools["image_search"].set_sample_context(data_source, sample_idx)

            # Reset per-sample match logs for text_search and visit
            if "text_search" in self.tools and hasattr(self.tools["text_search"], "reset_match_log"):
                self.tools["text_search"].reset_match_log()
            if "visit" in self.tools and hasattr(self.tools["visit"], "reset_match_log"):
                self.tools["visit"].reset_match_log()

            # Reset per-sample API usage tracker
            reset_api_usage()

            agent_kwargs = dict(kwargs)
            agent_kwargs.setdefault("application_id", str(uid))

            agent_start = time.perf_counter()
            if pil_images:
                result = await self.agent.run(
                    question=question,
                    answer=answer_nothink,
                    images=pil_images,
                    image_path=raw_images,
                    **agent_kwargs,
                )
            else:
                result = await self.agent.run(
                    question=question, answer=answer_nothink, **agent_kwargs
                )
            agent_time = time.perf_counter() - agent_start

            episode = self._convert_result_to_episode(result, task, uid)
            behavior_metrics = _extract_tool_behavior_metrics(result.get("messages", []))

            # Fast path: if killed by dynamic rollout early stopping,
            # skip reward/judge computation, set masked, and return immediately.
            stop_event = kwargs.get("stop_event")
            if (
                stop_event is not None
                and stop_event.is_set()
                and result.get("termination") == "stopped_early"
            ):
                episode.termination_reason = TerminationReason.UNKNOWN
                episode.metrics = {
                    "reward/outcome": 0.0,
                    "masked": 1.0,
                    "completed": 0.0,
                    **behavior_metrics,
                }
                episode.info["mask_reason"] = "stopped_early"
                logger.debug(
                    "Dynamic rollout: skipping reward/save for early-stopped episode %s", uid
                )
                return episode

            prediction = result.get("prediction", "")

            # Determine if accuracy reward is handled by reward_config.
            # When reward_config has accuracy/partial_accuracy, compute_accuracy_reward
            # handles the LLM judge internally — skip the legacy reward_function.
            has_accuracy_in_config = (
                "accuracy" in self.reward_config or "partial_accuracy" in self.reward_config
            )

            if not has_accuracy_in_config and self.reward_function is not None and prediction:
                reward_start = time.perf_counter()
                try:
                    if asyncio.iscoroutinefunction(self.reward_function):
                        reward_out = await self.reward_function(task, prediction)
                    else:
                        loop = asyncio.get_running_loop()
                        reward_out = await loop.run_in_executor(
                            self.executor,
                            lambda: self.reward_function(task, prediction),
                        )
                except Exception as err:  # noqa: BLE001
                    logger.debug("Reward function failed: %s", err)
                else:
                    if reward_out.is_correct is not None:
                        episode.is_correct = bool(reward_out.is_correct)
                    if isinstance(reward_out.metadata, dict):
                        reward_metadata = episode.info.setdefault("reward_metadata", {})
                        for key, value in reward_out.metadata.items():
                            if key not in reward_metadata:
                                reward_metadata[key] = value
                    if getattr(reward_out, "reward", None) is not None:
                        episode.info["reward_function_reward"] = float(
                            reward_out.reward
                        )
                finally:
                    reward_time = time.perf_counter() - reward_start

            # Check whether to mask the whole episode (skip masking during evaluation)
            is_val = getattr(self.agent.rollout_engine, "validate", False)
            if is_val:
                should_mask_episode, mask_reason = False, ""
            else:
                should_mask_episode, mask_reason = _should_mask_episode(result, episode)

            # Compute rewards based on reward_config
            messages = result.get("messages", [])
            format_reward_val = 0.0
            accuracy_reward_val = 0.0
            format_errors = []

            # Format reward
            if "format" in self.reward_config:
                fmt_weight = float(self.reward_config["format"])
                fmt_result = compute_format_reward(
                    messages=messages,
                    weight=fmt_weight,
                    tools_config=self.tools_config,
                    contract_penalty=float(self.reward_config.get("tool_contract_penalty", 0.2)),
                )
                format_reward_val = fmt_result["reward"]
                format_errors = fmt_result["format_errors"]

            # Accuracy reward — during validation, always use binary accuracy for
            # consistent metrics regardless of training reward mode.
            is_val = getattr(self.agent.rollout_engine, "validate", False)
            if is_val and "partial_accuracy" in self.reward_config:
                # Override: use binary accuracy for val to keep metrics comparable
                val_reward_config = {"accuracy": self.reward_config["partial_accuracy"]}
                has_partial = False
                has_accuracy = True
            else:
                val_reward_config = self.reward_config
                has_partial = "partial_accuracy" in self.reward_config
                has_accuracy = "accuracy" in self.reward_config

            if has_partial:
                # Three-tier judge: compute accuracy directly (async)
                acc_result = await compute_accuracy_reward(
                    messages=messages,
                    task=task,
                    prediction=prediction,
                    reward_config=val_reward_config,
                    executor=self.executor,
                )
                accuracy_reward_val = acc_result["reward"]
                episode.is_correct = acc_result["is_correct"]
                if acc_result.get("is_partial"):
                    episode.info["is_partial"] = True
                episode.info["accuracy_detail"] = acc_result.get("accuracy_detail", {})
            elif has_accuracy:
                # Binary mode (async)
                acc_result = await compute_accuracy_reward(
                    messages=messages,
                    task=task,
                    prediction=prediction,
                    reward_config=val_reward_config,
                    executor=self.executor,
                )
                accuracy_reward_val = acc_result["reward"]
                episode.is_correct = bool(acc_result["is_correct"])
                episode.info["accuracy_detail"] = acc_result.get("accuracy_detail", {})

            completed = 1.0 if result.get("termination") == "answer" else 0.0

            if should_mask_episode:
                episode.termination_reason = TerminationReason.UNKNOWN
                episode.metrics = {
                    "reward/outcome": 0.0,
                    "rounds": result.get("rounds", 0),
                    "masked": 1.0,
                    "completed": completed,
                    **behavior_metrics,
                }
                episode.info["mask_reason"] = mask_reason or result.get(
                    "termination", "unknown"
                )
                episode.info["format_errors"] = format_errors
            else:
                base_reward = format_reward_val + accuracy_reward_val
                global_efficiency_val = 0.0
                global_efficiency_metrics = {}
                if self.global_efficiency_reward is not None and not is_val:
                    tools_key = self.global_efficiency_reward.metric_tool_key
                    tools_value = behavior_metrics.get(
                        tools_key, behavior_metrics.get("tool_calls_expanded", behavior_metrics.get("tool_calls", 0))
                    )
                    try:
                        is_fully_correct = float(episode.is_correct) == 1.0
                    except (TypeError, ValueError):
                        is_fully_correct = bool(episode.is_correct)
                    ge_result = self.global_efficiency_reward.compute(
                        task_id=self._task_key(task, uid),
                        is_correct=is_fully_correct,
                        rounds=result.get("rounds", 0),
                        tools=tools_value,
                        episode_id=uid,
                    )
                    global_efficiency_val = ge_result.bonus
                    global_efficiency_metrics = ge_result.record

                final_reward = base_reward + global_efficiency_val
                for trajectory in episode.trajectories:
                    if not trajectory.steps:
                        continue
                    trajectory.reward = final_reward

                episode.metrics = {
                    "reward/outcome": final_reward,
                    "reward/accuracy": accuracy_reward_val,
                    "reward/format": format_reward_val,
                    "reward/outcome_base_global_efficiency": base_reward,
                    **global_efficiency_metrics,
                    "rounds": result.get("rounds", 0),
                    "masked": 0.0,
                    "completed": completed,
                    **behavior_metrics,
                }
                episode.info["format_errors"] = format_errors

            total_elapsed_before_save = time.perf_counter() - total_start
            timing = {
                "agent_time": round(agent_time, 4),
                "reward_time": round(reward_time, 4),
                "save_time": 0.0,
                "total_time": round(total_elapsed_before_save, 4),
            }

            # Collect image_search bbox match log (if tool was used)
            image_search_match_log = []
            if "image_search" in self.tools and hasattr(self.tools["image_search"], "get_match_log"):
                image_search_match_log = self.tools["image_search"].get_match_log()

            # Collect text_search and visit cache/online match logs
            text_search_match_log = []
            if "text_search" in self.tools and hasattr(self.tools["text_search"], "get_match_log"):
                text_search_match_log = self.tools["text_search"].get_match_log()

            visit_match_log = []
            if "visit" in self.tools and hasattr(self.tools["visit"], "get_match_log"):
                visit_match_log = self.tools["visit"].get_match_log()

            # Collect API usage for this sample
            api_usage = get_api_usage()

            # Save rollout data
            save_start = time.perf_counter()
            self._save_rollout_data(
                uid, data_id, result, episode, raw_images, answer_nothink, task, timing,
                image_search_match_log=image_search_match_log,
                text_search_match_log=text_search_match_log,
                visit_match_log=visit_match_log,
                api_usage=api_usage,
            )
            save_time = time.perf_counter() - save_start
            timing["save_time"] = round(save_time, 4)
            timing["total_time"] = round(time.perf_counter() - total_start, 4)

            # Minimal one-line log
            # Build compact usage summary (only non-zero counters)
            usage_parts = []
            for k in ("text_search", "image_search", "visit", "extract", "extract_tokens"):
                v = api_usage.get(k, 0)
                if v:
                    usage_parts.append(f"{k}={v}")
            usage_str = " ".join(usage_parts) if usage_parts else "none"

            logger.info(
                "task=%s correct=%s reward=%.1f masked=%.0f term=%s rounds=%d t=%.1fs agent=%.2fs reward=%.2fs total=%.2fs api=[%s]",
                uid,
                episode.is_correct,
                episode.metrics.get("reward/outcome", 0.0),
                episode.metrics.get("masked", 0.0),
                result.get("termination", "?"),
                result.get("rounds", 0),
                result.get("time_taken", 0.0),
                agent_time,
                reward_time,
                time.perf_counter() - total_start,
                usage_str,
            )

            return episode

        except Exception as exc:  # noqa: BLE001
            logger.warning("Workflow failed for task %s: %s", uid, exc)

            # Save error rollout data using same directory layout
            if self.rollout_save_dir is not None:
                total_elapsed = time.perf_counter() - total_start
                step_num = self._training_step
                mode = "val" if getattr(self.agent.rollout_engine, "validate", False) else "train"
                group_id, rollout_idx = self._parse_uid(uid)
                safe_group = str(group_id).replace("/", "_").replace(":", "_")
                safe_ridx = str(rollout_idx).replace("/", "_").replace(":", "_")
                sample_dir = os.path.join(
                    self.rollout_save_dir, mode, f"step_{step_num}",
                    f"sample_{safe_group}"
                )
                os.makedirs(sample_dir, exist_ok=True)
                save_path = os.path.join(sample_dir, f"rollout_{safe_ridx}.json")
                error_record = {
                    "uid": uid,
                    "group_id": group_id,
                    "rollout_idx": rollout_idx,
                    "step": step_num,
                    "id": data_id,
                    "messages": [],
                    "prediction": "",
                    "answer_nothink": answer_nothink,
                    "answer": task.get("answer", ""),
                    "question": task.get("question", ""),
                    "data_source": task.get("data_source", ""),
                    "data_source_path": task.get("data_source_path", ""),
                    "is_correct": False,
                    "termination": f"exception: {exc}",
                    "rounds": 0,
                    "time_taken": 0.0,
                    "reward/outcome": 0.0,
                    "masked": 1.0,
                    "completed": 0.0,
                    "mask_reason": f"exception: {exc}",
                    "reward_function_reward": None,
                    "token_usage": {},
                    "images": _serialize_images(task.get("images")),
                    "timing": {
                        "agent_time": round(agent_time, 4),
                        "reward_time": round(reward_time, 4),
                        "save_time": 0.0,
                        "total_time": round(total_elapsed, 4),
                    },
                }
                try:
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump(error_record, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            episode = Episode()
            episode.id = uid
            episode.task = task
            episode.termination_reason = TerminationReason.ERROR
            episode.is_correct = False
            episode.trajectories = [
                self._build_synthetic_error_trajectory(
                    task=task,
                    messages=None,
                    reason="Workflow failed before producing a valid model step.",
                    observation=f"[Workflow Error] {exc}",
                )
            ]
            episode.metrics = {
                "reward/outcome": 0.0,
                "masked": 1.0,
                "completed": 0.0,
                "error": 1.0,
            }
            episode.info = {"error": str(exc)}
            return episode

    def _convert_result_to_episode(self, result: dict, task: dict, uid: str) -> Episode:
        messages = result.get("messages", [])
        prediction = result.get("prediction", "")
        termination = result.get("termination", "unknown")
        rounds = result.get("rounds", 0)
        time_taken = result.get("time_taken", 0.0)

        trajectories: list[Trajectory] = []
        steps: list[Step] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg["role"] == "assistant":
                assistant_content = msg.get("content", "")
                action = _extract_action_from_response(assistant_content)
                observation = _get_next_observation(messages, i)
                step = Step(
                    chat_completions=messages[: i + 1],  # reference, no copy
                    model_response=assistant_content,
                    action=action,
                    observation=observation,
                    reward=0.0,
                )
                step.model_output = None
                if msg.get("step_error"):
                    step.info["step_error"] = True
                if _has_tool_error_observation(observation):
                    step.info["step_error"] = True
                steps.append(step)
            i += 1

        if steps:
            trajectory = Trajectory(
                name="deepresearch_agent",
                task=task,
                steps=steps,
                reward=0.0,
                info={},
            )
        else:
            trajectory = self._build_synthetic_error_trajectory(
                task=task,
                messages=messages,
                reason="Agent failed before producing a model response.",
                observation=f"[Agent Error] {termination}",
            )
        trajectories.append(trajectory)

        answer_text = task.get("answer", "")
        is_correct = _evaluate_answer(prediction, answer_text) if answer_text else False

        episode = Episode()
        episode.id = uid
        episode.task = task
        episode.termination_reason = _map_termination_reason(termination)
        episode.is_correct = is_correct
        episode.trajectories = trajectories
        episode.metrics = {}
        episode.info = {
            "rounds": rounds,
            "time_taken": time_taken,
            "prediction": prediction,
            "answer": answer_text,
            "token_usage": result.get("token_usage", {}),
        }
        return episode

    def reset(self, task: dict | None = None, uid: str | None = None):
        return

    def is_multithread_safe(self) -> bool:
        return True


__all__ = ["DeepResearchWorkflow"]
