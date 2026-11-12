"""
Accuracy reward for Mars Agentic Search.

Supports two modes:
- "accuracy" (binary): correct=1.0, incorrect=0.0 (original behavior)
- "partial_accuracy" (three-tier): correct=1.0, partial=0.5, incorrect=0.0

The binary mode reuses RewardDeepResearchFn from rllm.rewards.
The three-tier mode uses a new LLM judge prompt with correct/partial/incorrect ratings.

Both modes are async to avoid blocking the event loop during training.
"""

import asyncio
import json
import logging
import os
import random
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx

from rllm.rewards.deepresearch_reward import RewardDeepResearchFn
from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Three-tier judge prompt
# ---------------------------------------------------------------------------

def _build_three_tier_prompt(question: str, answer: str, prediction: str) -> str:
    """Build the three-tier judge prompt (correct/partial/incorrect)."""
    multi_answer_note = ""
    if "||" in answer:
        multi_answer_note = (
            '\nNote: The ground truth contains multiple valid answers '
            'separated by "||" — the prediction is correct if it matches ANY one.\n'
        )
    return (
        "Judge whether the prediction matches the ground truth answer.\n\n"
        "[Question]\n{question}\n\n"
        "[Ground Truth]\n{reference_answer}\n"
        "{multi_answer_note}\n"
        "[Prediction]\n{prediction}\n\n"
        "Rate:\n"
        "- correct: The GT name appears verbatim in the prediction, "
        "OR the prediction and GT are the same named entity with only cosmetic differences "
        "(articles, capitalization, extra words that narrow but do not rename the entity). "
        "The GT name embedded inside a sentence still counts as correct. "
        "Renaming or paraphrasing the GT entity does NOT count.\n"
        "- partial: The prediction is topically related to the GT but fails the correct test. "
        "This includes: the prediction is a hypernym of the GT; "
        "the prediction is a different entity in the same domain; "
        "the prediction rephrases the GT using different words that are not the GT name; "
        "the prediction names a related but distinct entity type. "
        "When uncertain between partial and incorrect, choose partial.\n"
        "- incorrect: The prediction is topically unrelated to the GT or directly contradicts it.\n\n"
        "Output:\n"
        "correct: [yes/partial/no]\n"
        "reasoning: [brief explanation]"
    ).format(
        question=question,
        reference_answer=answer,
        multi_answer_note=multi_answer_note,
        prediction=prediction,
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_three_tier_judgment(text: str) -> str | None:
    """Parse three-tier judge output. Returns 'correct', 'partial', or 'incorrect'."""
    lt = text.lower()
    m = re.search(r"correct:\s*(yes|partial|no)\b", lt)
    if m:
        val = m.group(1)
        if val == "yes":
            return "correct"
        elif val == "partial":
            return "partial"
        else:
            return "incorrect"
    return None


# ---------------------------------------------------------------------------
# HTTP helpers (shared with deepresearch_reward.py pattern)
# ---------------------------------------------------------------------------

def _get_requests_proxies() -> dict[str, str | None] | None:
    proxy_value = os.getenv("TOOL_HTTPS_PROXY")
    if proxy_value is None:
        return None
    proxy_value = proxy_value.strip()
    if not proxy_value or proxy_value.lower() == "none":
        return {"http": None, "https": None}
    return {"http": proxy_value, "https": proxy_value}


def _load_judge_api_configs(default_model: str) -> list[dict[str, str]]:
    """Load one or more OpenAI-compatible judge configs from environment."""
    fallback_key = (
        os.getenv("JUDGE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("TOGETHER_AI_API_KEY")
    )
    fallback_base_url = os.getenv("JUDGE_BASE_URL")
    configs: list[dict[str, str]] = []

    def split_env(name: str) -> list[str]:
        return [item.strip() for item in os.getenv(name, "").split(",") if item.strip()]

    def add(model: str | None, base_url: str | None, api_key: str | None) -> None:
        if not model or not api_key:
            return
        resolved_base_url = base_url
        if not resolved_base_url:
            if os.getenv("TOGETHER_AI_API_KEY") and api_key == os.getenv("TOGETHER_AI_API_KEY"):
                resolved_base_url = "https://api.together.xyz/v1"
            else:
                resolved_base_url = "https://api.openai.com/v1"
        configs.append(
            {
                "model": model,
                "base_url": resolved_base_url,
                "api_key": api_key,
            }
        )

    models = split_env("JUDGE_MODELS")
    base_urls = split_env("JUDGE_BASE_URLS")
    api_keys = split_env("JUDGE_API_KEYS")
    if models:
        for idx, model in enumerate(models):
            add(
                model,
                base_urls[idx] if idx < len(base_urls) else fallback_base_url,
                api_keys[idx] if idx < len(api_keys) else fallback_key,
            )
    else:
        add(default_model, fallback_base_url, fallback_key)
    return configs


def _run_with_retries_sync(
    func: Callable[[], T], attempts: int = 3, delay: float = 1.0
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            if delay > 0:
                time.sleep(delay * attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError("run_with_retries_sync executed without performing any attempts")


async def _async_run_with_retries(
    func: Callable[[], T], attempts: int = 3, delay: float = 1.0
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            return await asyncio.to_thread(func)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            if delay > 0:
                await asyncio.sleep(delay * attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError("async_run_with_retries executed without performing any attempts")


def _run_async_or_sync(
    coro: Awaitable[T], fallback: Callable[[], T], *, fallback_tag: str
) -> tuple[T, str]:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro), "async"
    return fallback(), "sync_fallback"


# ---------------------------------------------------------------------------
# Three-tier judge class (cached instance)
# ---------------------------------------------------------------------------

_judge_instance: "ThreeTierJudge | None" = None
_binary_judge_instance: "RewardDeepResearchFn | None" = None


class ThreeTierJudge:
    """LLM judge that outputs correct/partial/incorrect."""

    def __init__(self):
        self.base_url: str | None = os.getenv("JUDGE_BASE_URL")
        raw_vllm_urls = os.getenv("JUDGE_URL", "")
        self.vllm_urls: list[str] = [
            item.strip() for item in raw_vllm_urls.split(",") if item.strip()
        ]
        self.is_vllm: bool = bool(self.vllm_urls)
        self.api_key: str | None = (
            os.getenv("JUDGE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("TOGETHER_AI_API_KEY")
        )
        self.model: str = os.getenv(
            "JUDGE_MODEL", os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")
        )
        self.api_configs = _load_judge_api_configs(self.model)

        if not self.is_vllm and self.base_url is None:
            if os.getenv("TOGETHER_AI_API_KEY") and self.api_key == os.getenv(
                "TOGETHER_AI_API_KEY"
            ):
                self.base_url = "https://api.together.xyz/v1"
                if "gpt-" in self.model.lower():
                    self.model = os.getenv(
                        "JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo"
                    )
            else:
                self.base_url = "https://api.openai.com/v1"

        if self.is_vllm and "gpt-" in self.model.lower():
            self.model = os.getenv("JUDGE_MODEL", "Qwen/Qwen2.5-7B-Instruct-Turbo")

        self._client: httpx.Client | None = None
        self._judge_proxies = _get_requests_proxies()
        self.retry_attempts = int(os.getenv("JUDGE_RETRY_ATTEMPTS", "8"))
        self.retry_delay = float(os.getenv("JUDGE_RETRY_DELAY", "1.0"))
        self.timeout = float(os.getenv("JUDGE_TIMEOUT", "300"))

        if self.api_configs or self.is_vllm:
            try:
                self._client = httpx.Client(timeout=60.0)
            except Exception:
                self._client = None

    def _select_vllm_url(self) -> str | None:
        if not self.vllm_urls:
            return None
        return random.choice(self.vllm_urls)

    def _select_api_config(self) -> dict[str, str] | None:
        if not self.api_configs:
            return None
        return random.choice(self.api_configs)

    def judge(self, question: str, answer: str, prediction: str) -> tuple[str, dict]:
        """
        Run three-tier judge.

        Returns:
            (tier, metadata) where tier is 'correct', 'partial', or 'incorrect'.
            Falls back to exact match if judge is unavailable.
        """
        prompt = _build_three_tier_prompt(question, answer, prediction)

        # Exact match fallback
        pred_normalized = (prediction or "").strip().lower()
        refs = [a.strip().lower() for a in str(answer).split("||")]
        exact_match = pred_normalized in refs
        if exact_match:
            return "correct", {
                "judge_used": False,
                "exact_match": True,
                "tier": "correct",
                "judge_skip_reason": "exact_match",
            }

        if self._client is None:
            logger.debug("ThreeTierJudge: no client, falling back to exact match")
            tier = "correct" if exact_match else "incorrect"
            return tier, {"judge_used": False, "exact_match": exact_match, "tier": tier}

        if self.is_vllm and self.vllm_urls:
            tier, meta = self._judge_vllm(prompt, exact_match)
            return tier, meta
        elif self.api_configs:
            tier, meta = self._judge_openai(prompt, exact_match)
            return tier, meta

        # No judge available
        tier = "correct" if exact_match else "incorrect"
        return tier, {"judge_used": False, "exact_match": exact_match, "tier": tier}

    def _judge_vllm(self, prompt: str, exact_match: bool) -> tuple[str, dict]:
        """Judge via vLLM endpoint."""
        try:
            import requests
        except ImportError:
            tier = "correct" if exact_match else "incorrect"
            return tier, {"judge_used": False, "exact_match": exact_match, "tier": tier, "error": "requests_missing"}

        url = self._select_vllm_url()
        if not re.search(r"/v1/chat/completions/?$", url):
            url = f"{url.rstrip('/')}/v1/chat/completions"

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 800,
        }
        headers = {"Content-Type": "application/json"}
        attempt_counter: dict[str, int] = {"value": 0}

        def send_request() -> "requests.Response":
            attempt_counter["value"] += 1
            resp = requests.post(
                url=url, headers=headers, json=payload,
                timeout=self.timeout, proxies=self._judge_proxies,
            )
            resp.raise_for_status()
            return resp

        try:
            async_coro = _async_run_with_retries(
                send_request, attempts=self.retry_attempts, delay=self.retry_delay,
            )
            resp, judge_execution_mode = _run_async_or_sync(
                async_coro,
                lambda: _run_with_retries_sync(send_request, attempts=self.retry_attempts, delay=self.retry_delay),
                fallback_tag="ThreeTierJudge/vLLM",
            )
        except Exception as exc:
            tier = "correct" if exact_match else "incorrect"
            return tier, {"judge_used": False, "exact_match": exact_match, "tier": tier, "error": str(exc)}

        try:
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            tier = "correct" if exact_match else "incorrect"
            return tier, {"judge_used": False, "exact_match": exact_match, "tier": tier, "error": str(exc)}

        judgment_text = self._extract_content(data)
        tier = parse_three_tier_judgment(judgment_text or "")
        if tier is None:
            tier = "correct" if exact_match else "incorrect"
            return tier, {
                "judge_used": True, "judge_decided": False, "exact_match": exact_match,
                "tier": tier, "judgment_preview": (judgment_text or "")[:200],
            }

        return tier, {
            "judge_used": True, "judge_decided": True, "exact_match": exact_match,
            "tier": tier, "judgment_preview": (judgment_text or "")[:200],
        }

    def _judge_openai(self, prompt: str, exact_match: bool) -> tuple[str, dict]:
        """Judge via OpenAI-compatible API."""
        for attempt in range(self.retry_attempts):
            api_config = self._select_api_config()
            if api_config is None:
                break
            try:
                payload = {
                    "model": api_config["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": 800,
                }
                resp = self._client.post(
                    url=f"{api_config['base_url'].rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_config['api_key']}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                judgment_text = self._extract_content(data)

                tier = parse_three_tier_judgment(judgment_text or "")
                if tier is not None:
                    return tier, {
                        "judge_used": True, "judge_decided": True,
                        "exact_match": exact_match, "tier": tier,
                        "judgment_preview": (judgment_text or "")[:200],
                    }
            except Exception as exc:
                if attempt + 1 >= 5:
                    logger.debug(
                        "ThreeTierJudge/OpenAI error (attempt %d, model=%s): %s",
                        attempt + 1,
                        api_config["model"],
                        exc,
                    )
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        tier = "correct" if exact_match else "incorrect"
        return tier, {"judge_used": False, "exact_match": exact_match, "tier": tier}

    @staticmethod
    def _extract_content(data: dict) -> str | None:
        choices = data.get("choices") or [{}]
        first = choices[0] if choices else {}
        msg = first.get("message", {}) if isinstance(first, dict) else {}
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if not content and isinstance(data, dict):
            content = data.get("content", "")
        return content or None


def _get_judge() -> ThreeTierJudge:
    """Get or create the cached ThreeTierJudge instance."""
    global _judge_instance
    if _judge_instance is None:
        _judge_instance = ThreeTierJudge()
    return _judge_instance


def _get_binary_judge() -> RewardDeepResearchFn:
    """Get or create the cached binary RewardDeepResearchFn instance."""
    global _binary_judge_instance
    if _binary_judge_instance is None:
        _binary_judge_instance = RewardDeepResearchFn(RewardConfig())
    return _binary_judge_instance


# ---------------------------------------------------------------------------
# Accuracy reward computation (async)
# ---------------------------------------------------------------------------

async def compute_accuracy_reward(
    *,
    messages: list[dict],
    task: dict,
    prediction: str,
    reward_config: dict,
    executor=None,
) -> dict:
    """
    Compute accuracy reward based on reward_config.

    reward_config keys:
      - "accuracy": weight (float) → binary correct/incorrect via RewardDeepResearchFn
      - "partial_accuracy": weight (float) → three-tier correct/partial/incorrect

    Returns dict with:
      - reward: float (total accuracy reward)
      - is_correct: bool (True only if fully correct)
      - is_partial: bool (True if partially correct, only for partial_accuracy mode)
      - accuracy_detail: dict with mode-specific metadata
    """
    # Determine mode
    has_accuracy = "accuracy" in reward_config
    has_partial = "partial_accuracy" in reward_config

    if has_accuracy and has_partial:
        raise ValueError(
            "reward config cannot have both 'accuracy' and 'partial_accuracy'. "
            "Use 'accuracy' for binary or 'partial_accuracy' for three-tier."
        )

    if not has_accuracy and not has_partial:
        # No accuracy reward configured
        return {
            "reward": 0.0,
            "is_correct": False,
            "is_partial": False,
            "accuracy_detail": {"mode": "none"},
        }

    answer = task.get("answer")
    question = str(task.get("question", ""))

    if has_accuracy:
        # Binary mode: use cached RewardDeepResearchFn (async)
        weight = float(reward_config["accuracy"])
        fn = _get_binary_judge()
        reward_input = RewardInput(task_info=task, action=prediction)

        # Run in executor to avoid blocking the event loop
        if executor is not None:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(executor, lambda: fn(reward_input))
        else:
            result = await asyncio.to_thread(fn, reward_input)

        is_correct = bool(result.is_correct) if result.is_correct is not None else False
        return {
            "reward": weight if is_correct else 0.0,
            "is_correct": is_correct,
            "is_partial": False,
            "accuracy_detail": {
                "mode": "accuracy",
                "weight": weight,
                "is_correct": is_correct,
                "metadata": result.metadata,
            },
        }

    # Partial accuracy mode: three-tier judge (async)
    weight = float(reward_config["partial_accuracy"])
    judge = _get_judge()

    # Run judge in executor to avoid blocking the event loop
    if executor is not None:
        loop = asyncio.get_running_loop()
        tier, judge_meta = await loop.run_in_executor(
            executor, lambda: judge.judge(question, str(answer), prediction)
        )
    else:
        tier, judge_meta = await asyncio.to_thread(
            judge.judge, question, str(answer), prediction
        )

    is_correct = tier == "correct"
    is_partial = tier == "partial"

    if tier == "correct":
        acc_reward = weight
    elif tier == "partial":
        acc_reward = weight / 2.0
    else:
        acc_reward = 0.0

    return {
        "reward": acc_reward,
        "is_correct": is_correct,
        "is_partial": is_partial,
        "accuracy_detail": {
            "mode": "partial_accuracy",
            "weight": weight,
            "tier": tier,
            "judge_meta": judge_meta,
        },
    }
