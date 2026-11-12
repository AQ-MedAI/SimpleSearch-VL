"""DAPO group-signal helpers.

Keep this module dependency-light so the sampling policy can be unit-tested
without importing the full Ray/verl trainer stack.
"""

from __future__ import annotations

import math
from typing import Iterable


def normalize_dapo_signal_source(value) -> str:
    if value is None or value == "":
        return "reward"
    source = str(value).strip().lower().replace("-", "_")
    aliases = {
        "reward": "reward",
        "reward_std": "reward",
        "outcome": "reward",
        "outcome_reward": "reward",
        "correct": "correctness",
        "correctness": "correctness",
        "is_correct": "correctness",
        "accuracy": "correctness",
        "acc": "correctness",
    }
    if source not in aliases:
        raise ValueError(
            "dapo_sampling.signal_source must be one of "
            "reward or correctness; got %r" % value
        )
    return aliases[source]


def _to_finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _to_correctness(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "correct", "yes", "1"}:
            return True
        if normalized in {"false", "incorrect", "no", "0"}:
            return False
    number = _to_finite_float(value)
    if number is None:
        return None
    return number > 0.5


def dapo_group_has_signal(values: Iterable, reward_std_threshold: float, signal_source="reward") -> bool:
    source = normalize_dapo_signal_source(signal_source)
    if source == "correctness":
        outcomes = []
        for value in values:
            outcome = _to_correctness(value)
            if outcome is not None:
                outcomes.append(outcome)
        if len(outcomes) < 2:
            return False
        return any(outcomes) and not all(outcomes)

    rewards = []
    for value in values:
        reward = _to_finite_float(value)
        if reward is not None:
            rewards.append(reward)
    if len(rewards) < 2:
        return False
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    return math.sqrt(variance) > float(reward_std_threshold)
