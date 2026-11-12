"""Workflow-level persistent global efficiency reward.

The reward compares a correct rollout against historical successful rollouts of
that same source sample. It only gives positive bonuses for being more efficient;
it never penalizes slower or more tool-heavy successful rollouts.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class GlobalEfficiencyResult:
    bonus: float
    round_bonus: float
    tool_bonus: float
    has_reference: bool
    ref_rounds: float | None
    ref_tools: float | None
    rounds: float
    tools: float
    record: dict[str, Any]


class GlobalEfficiencyReward:
    """Persistent per-sample efficiency bonus for workflow rewards."""

    def __init__(
        self,
        *,
        state_path: str,
        reference: str = "mean",
        round_weight: float = 0.1,
        tool_weight: float = 0.1,
        metric_tool_key: str = "tool_calls_expanded",
    ) -> None:
        self.state_path = str(state_path or "")
        self.reference = str(reference or "mean").lower()
        if self.reference not in {"mean", "min"}:
            raise ValueError(f"Unsupported global efficiency reference: {reference}")
        self.round_weight = float(round_weight or 0.0)
        self.tool_weight = float(tool_weight or 0.0)
        self.metric_tool_key = str(metric_tool_key or "tool_calls_expanded")

    def compute(
        self,
        *,
        task_id: str,
        is_correct: bool,
        rounds: Any,
        tools: Any,
        episode_id: str = "",
    ) -> GlobalEfficiencyResult:
        rounds_f = self._safe_float(rounds)
        tools_f = self._safe_float(tools)
        task_key = str(task_id or episode_id or "unknown")
        if not self.state_path:
            return self._result(0.0, 0.0, 0.0, False, None, None, rounds_f, tools_f)

        with self._read_state_locked() as state:
            ref_record = state.get(task_key)
            has_reference = (
                isinstance(ref_record, dict)
                and int(ref_record.get("success_count", 0) or 0) > 0
            )
            ref_rounds = self._reference_value(ref_record, "round") if has_reference else None
            ref_tools = self._reference_value(ref_record, "tool") if has_reference else None

        round_bonus = 0.0
        tool_bonus = 0.0
        if is_correct and has_reference:
            if self.round_weight > 0 and ref_rounds is not None and rounds_f < ref_rounds:
                round_bonus = float(self.round_weight)
            if self.tool_weight > 0 and ref_tools is not None and tools_f < ref_tools:
                tool_bonus = float(self.tool_weight)

        bonus = float(round_bonus + tool_bonus)
        return self._result(
            bonus, round_bonus, tool_bonus, has_reference,
            ref_rounds, ref_tools, rounds_f, tools_f, task_key,
            str(episode_id or ""), bool(is_correct),
        )


    def update_many(self, records: list[dict[str, Any]]) -> int:
        if not self.state_path:
            return 0
        updates = []
        for record in records:
            if not isinstance(record, dict):
                continue
            if not bool(record.get("is_correct", False)):
                continue
            task_key = str(record.get("task_id") or record.get("episode_id") or "")
            if not task_key:
                continue
            rounds = self._safe_float(record.get("rounds"), default=None)
            tools = self._safe_float(record.get("tools"), default=None)
            if rounds is None or tools is None:
                continue
            updates.append((task_key, rounds, tools, str(record.get("episode_id") or "")))
        if not updates:
            return 0
        with self._locked_state() as state:
            for task_key, rounds, tools, episode_id in updates:
                state[task_key] = self._updated_record(
                    state.get(task_key), rounds, tools, episode_id
                )
        return len(updates)

    @contextlib.contextmanager
    def _read_state_locked(self):
        state_dir = os.path.dirname(self.state_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
        lock_path = f"{self.state_path}.lock"
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            state = self._load_state()
            yield state
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @contextlib.contextmanager
    def _locked_state(self):
        state_dir = os.path.dirname(self.state_path)
        if state_dir:
            os.makedirs(state_dir, exist_ok=True)
        lock_path = f"{self.state_path}.lock"
        with open(lock_path, "w", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            state = self._load_state()
            yield state
            self._save_state(state)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> dict[str, dict[str, Any]]:
        if not self.state_path or not os.path.exists(self.state_path):
            return {}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return {}
        if isinstance(payload, dict) and isinstance(payload.get("samples"), dict):
            return payload["samples"]
        return payload if isinstance(payload, dict) else {}

    def _save_state(self, state: dict[str, dict[str, Any]]) -> None:
        payload = {
            "version": 2,
            "reference": self.reference,
            "metric_tool_key": self.metric_tool_key,
            "samples": state,
        }
        tmp_path = f"{self.state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp_path, self.state_path)

    def _reference_value(self, record: dict[str, Any] | None, prefix: str) -> float | None:
        if not isinstance(record, dict):
            return None
        if self.reference == "min":
            return self._safe_float(record.get(f"{prefix}_min"), default=None)
        return self._safe_float(record.get(f"{prefix}_mean"), default=None)

    @staticmethod
    def _safe_float(value: Any, default: float | None = 0.0) -> float | None:
        try:
            out = float(value)
        except Exception:
            return default
        return out if math.isfinite(out) else default

    def _updated_record(
        self,
        record: dict[str, Any] | None,
        rounds: float,
        tools: float,
        episode_id: str,
    ) -> dict[str, Any]:
        if not isinstance(record, dict):
            record = {
                "success_count": 0,
                "round_sum": 0.0,
                "tool_sum": 0.0,
                "round_min": None,
                "tool_min": None,
                "last_episode_id": "",
            }
        count = int(record.get("success_count", 0) or 0) + 1
        round_sum = float(record.get("round_sum", 0.0) or 0.0) + float(rounds)
        tool_sum = float(record.get("tool_sum", 0.0) or 0.0) + float(tools)
        prev_round_min = record.get("round_min")
        prev_tool_min = record.get("tool_min")
        round_min = rounds if prev_round_min is None else min(float(prev_round_min), rounds)
        tool_min = tools if prev_tool_min is None else min(float(prev_tool_min), tools)
        record.update({
            "success_count": count,
            "round_sum": round_sum,
            "tool_sum": tool_sum,
            "round_mean": round_sum / count,
            "tool_mean": tool_sum / count,
            "round_min": round_min,
            "tool_min": tool_min,
            "last_episode_id": episode_id,
        })
        return record

    def _result(
        self,
        bonus: float,
        round_bonus: float,
        tool_bonus: float,
        has_reference: bool,
        ref_rounds: float | None,
        ref_tools: float | None,
        rounds: float,
        tools: float,
        task_key: str = "",
        episode_id: str = "",
        should_update: bool = False,
    ) -> GlobalEfficiencyResult:
        record = {
            "reward/global_efficiency": float(bonus),
            "global_efficiency/reference": self.reference,
            "global_efficiency/task_id": str(task_key or ""),
            "global_efficiency/episode_id": str(episode_id or ""),
            "global_efficiency/update_state": 1.0 if should_update else 0.0,
            "global_efficiency/has_reference": 1.0 if has_reference else 0.0,
            "global_efficiency/ref_rounds": ref_rounds if ref_rounds is not None else 0.0,
            "global_efficiency/ref_tools": ref_tools if ref_tools is not None else 0.0,
            "global_efficiency/rounds": float(rounds),
            "global_efficiency/tools": float(tools),
            "global_efficiency/round_bonus": float(round_bonus),
            "global_efficiency/tool_bonus": float(tool_bonus),
        }
        return GlobalEfficiencyResult(
            bonus=float(bonus),
            round_bonus=float(round_bonus),
            tool_bonus=float(tool_bonus),
            has_reference=has_reference,
            ref_rounds=ref_rounds,
            ref_tools=ref_tools,
            rounds=float(rounds),
            tools=float(tools),
            record=record,
        )
