import asyncio
import json
import math
import os
import random
import threading
import uuid
from collections import Counter, defaultdict
from functools import reduce
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import (
    AdvantageEstimator,
    agg_loss,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
)
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.debug import marked_timer
from simple_agentic_search.reward.global_efficiency_reward import GlobalEfficiencyReward


from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout.verl_engine import VerlEngine
from rllm.trainer.verl.mastered_blacklist_sampler import MasteredBlacklistSampler
from rllm.trainer.verl.dapo_signal import dapo_group_has_signal, normalize_dapo_signal_source
from rllm.utils import colorful_print
from rllm.utils.episode_logger import EpisodeLogger
from rllm.workflows.workflow import TerminationReason


class AgentWorkflowPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        workflow_class=None,
        workflow_args=None,
    ):
        self.mastered_filter_cfg = config.rllm.get("mastered_sample_filter", {})
        self.mastered_filter_enabled = bool(self.mastered_filter_cfg.get("enable", False))
        self.mastered_sampler = None
        self._mastered_filter_id_warning_logged = False
        self._mastered_filter_log = []

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        self.workflow_class = workflow_class
        self.workflow_args = workflow_args or {}
        self._validate_config()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _validate_config(self):
        assert (
            self.workflow_class is not None
        ), "workflow_class is required for agent workflow trainer"
        assert (
            self.config.actor_rollout_ref.hybrid_engine is True
        ), "Only hybrid engine is supported"
        assert (
            self.config.actor_rollout_ref.rollout.mode == "async"
        ), "Only async rollout mode is supported"
        assert (
            self.use_rm is False
        ), "Reward models are not supported. Rewards should be assigned using a reward function in the workflow or environment."
        if self.config.rllm.rejection_sample.multiplier != 1:
            assert (
                self.config.rllm.rejection_sample.enable is True
            ), "rejection sampling is disabled, but rejection_sample.multiplier is not 1"

        dapo_cfg = self.config.rllm.get("dapo_sampling", None)
        if dapo_cfg is not None and dapo_cfg.get("enable", False):
            assert (
                self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
                >= self.config.data.train_batch_size
            ), "dapo_sampling requires data.gen_batch_size >= data.train_batch_size"

        # TODO: revisit whether this is now supported by Verl
        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise NotImplementedError("REMAX is not supported yet")

    def init_workers(self):
        super().init_workers()

        rollout_engine = VerlEngine(
            config=self.config,
            rollout_manager=self.async_rollout_manager,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        # Create episode logger if enabled in config
        episode_logger = None
        if self.config.trainer.get("log_episodes", False):
            # Get episode log directory from config, default to "logs/my_project/my_experiment"
            episode_log_dir = self.config.trainer.get(
                "episode_log_dir",
                f"logs/{self.config.trainer.project_name}/{self.config.trainer.experiment_name}",
            )
            episode_logger = EpisodeLogger(
                base_dir=episode_log_dir, subdirectory="episodes"
            )

        self.agent_execution_engine = AgentWorkflowEngine(
            workflow_cls=self.workflow_class,
            workflow_args=self.workflow_args,
            rollout_engine=rollout_engine,
            config=self.config,
            n_parallel_tasks=self.config.rllm.workflow.n_parallel_tasks,
            n_parallel_tools=self.config.rllm.workflow.get("n_parallel_tools"),
            retry_limit=self.config.rllm.workflow.retry_limit,
            episode_logger=episode_logger,
        )

        # init workflow workers
        asyncio.run_coroutine_threadsafe(
            self.agent_execution_engine.initialize_pool(), self._loop
        ).result()

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler=None):
        if self.mastered_filter_enabled and train_sampler is None:
            if train_dataset is None:
                from verl.trainer.main_ppo import create_rl_dataset
                train_dataset = create_rl_dataset(
                    self.config.data.train_files,
                    self.config.data,
                    self.tokenizer,
                    self.processor,
                    max_samples=self.config.data.get("train_max_samples", -1),
                )
            seed = self.config.data.get("seed", 0)
            shuffle = bool(self.config.data.get("shuffle", False))
            train_sampler = MasteredBlacklistSampler(train_dataset, shuffle=shuffle, seed=seed)
            self.mastered_sampler = train_sampler
        super()._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _save_checkpoint(self):
        super()._save_checkpoint()
        self._flush_mastered_filter_log()
        if not self.mastered_filter_enabled or self.mastered_sampler is None:
            return
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        state_path = os.path.join(local_global_step_folder, "mastered_sample_filter_state.json")
        with open(state_path, "w") as f:
            json.dump(self.mastered_sampler.state_dict(), f)

    def _flush_mastered_filter_log(self):
        if not self._mastered_filter_log:
            return
        log_path = os.path.join(
            self.config.trainer.default_local_dir, "mastered_filter_log.jsonl"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            for record in self._mastered_filter_log:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._mastered_filter_log.clear()

    @staticmethod
    def _metric_value(metric_dict, *keys, default=0.0):
        if not isinstance(metric_dict, dict):
            return default
        for key in keys:
            value = _to_finite_float(metric_dict.get(key))
            if value is not None:
                return value
        return default

    def _collect_rollout_overview_metrics(self, episode_unique_batch, selection_info=None) -> dict:
        non_tensors = episode_unique_batch.non_tensor_batch
        episode_metrics = non_tensors.get("metrics", [])
        is_correct = np.asarray(non_tensors.get("is_correct", []), dtype=bool)
        episode_ids = [str(x) for x in non_tensors.get("episode_ids", [])]
        task_ids = non_tensors.get("task_ids", None)
        if task_ids is None:
            task_ids = [eid.rsplit(":", 1)[0] if ":" in eid else eid for eid in episode_ids]
        task_ids = [str(x) for x in task_ids]

        rewards = []
        format_values = []
        rounds = []
        time_taken = []
        tool_calls = []
        tool_calls_expanded = []
        global_efficiency_bonus = []
        for metric_dict, correct in zip(episode_metrics, is_correct, strict=False):
            reward = _get_reward_from_metrics(metric_dict)
            rewards.append(float(reward if reward is not None else (1.0 if correct else 0.0)))
            format_values.append(self._metric_value(metric_dict, "reward/format", "format_reward"))
            rounds.append(self._metric_value(metric_dict, "rounds", "num_rounds"))
            time_taken.append(self._metric_value(metric_dict, "time_taken", "elapsed_time", "duration"))
            tool_calls.append(self._metric_value(metric_dict, "tool_calls", "num_tool_calls", "total_tool_calls"))
            tool_calls_expanded.append(self._metric_value(metric_dict, "tool_calls_expanded", "total_tool_calls_expanded"))
            global_efficiency_bonus.append(self._metric_value(metric_dict, "reward/global_efficiency"))

        def mean(values):
            return float(np.mean(values)) if values else 0.0

        selected = discarded = generated = 0
        if isinstance(selection_info, dict):
            selected = len(selection_info.get("selected_episode_ids", []) or [])
            discarded = len(selection_info.get("discarded_episode_ids", []) or [])
            generated = selected + discarded
        else:
            selected = len(episode_ids)
            generated = len(episode_ids)

        unique_samples = len(set(task_ids))
        selected_rate = (selected * 100.0 / generated) if generated > 0 else 0.0
        metrics = {
            "rollout_overview/total_rollouts": float(len(episode_ids)),
            "rollout_overview/unique_samples": float(unique_samples),
            "rollout_overview/avg_reward": mean(rewards),
            "rollout_overview/correct_rate": mean(is_correct.astype(float).tolist()) * 100.0 if len(is_correct) else 0.0,
            "rollout_overview/format_pass_rate": mean([1.0 if x > 0 else 0.0 for x in format_values]) * 100.0,
            "rollout_overview/avg_tool_calls": mean(tool_calls),
            "rollout_overview/avg_tool_calls_expanded": mean(tool_calls_expanded),
            "rollout_overview/avg_rounds": mean(rounds),
            "rollout_overview/avg_time_taken": mean(time_taken),
            "rollout_overview/selected_rollouts": float(selected),
            "rollout_overview/discarded_rollouts": float(discarded),
            "rollout_overview/generated_rollouts": float(generated),
            "rollout_overview/selected_rate": float(selected_rate),
            "rollout_overview/avg_global_efficiency_bonus": mean(global_efficiency_bonus),
            "rollout_overview/global_efficiency_nonzero_rate": mean([1.0 if x != 0 else 0.0 for x in global_efficiency_bonus]) * 100.0,
        }
        return metrics


    def _update_global_efficiency_state_after_step(self, episode_unique_batch) -> int:
        reward_cfg = self.workflow_args.get("reward_config") or {}
        if not isinstance(reward_cfg, dict):
            return 0
        ge_cfg = reward_cfg.get("global_efficiency") or {}
        if not isinstance(ge_cfg, dict) or not bool(ge_cfg.get("enable", False)):
            return 0
        state_path = ge_cfg.get("state_path")
        if not state_path:
            return 0

        non_tensors = episode_unique_batch.non_tensor_batch
        metrics_list = non_tensors.get("metrics", [])
        is_correct = non_tensors.get("is_correct", [])
        episode_ids = [str(x) for x in non_tensors.get("episode_ids", [])]
        task_ids = non_tensors.get("task_ids", None)
        if task_ids is None:
            task_ids = [eid.rsplit(":", 1)[0] if ":" in eid else eid for eid in episode_ids]
        task_ids = [str(x) for x in task_ids]

        records = []
        for idx, metrics_dict in enumerate(metrics_list):
            if idx >= len(is_correct):
                break
            try:
                correct = float(is_correct[idx]) == 1.0
            except (TypeError, ValueError):
                correct = bool(is_correct[idx])
            if not correct:
                continue
            metrics_dict = metrics_dict if isinstance(metrics_dict, dict) else {}
            task_id = metrics_dict.get("global_efficiency/task_id") or (task_ids[idx] if idx < len(task_ids) else "")
            episode_id = metrics_dict.get("global_efficiency/episode_id") or (episode_ids[idx] if idx < len(episode_ids) else "")
            rounds = metrics_dict.get("global_efficiency/rounds", metrics_dict.get("rounds"))
            tools = metrics_dict.get("global_efficiency/tools")
            if tools is None:
                tools = metrics_dict.get("tool_calls_expanded", metrics_dict.get("tool_calls", 0))
            records.append({
                "task_id": task_id,
                "episode_id": episode_id,
                "is_correct": correct,
                "rounds": rounds,
                "tools": tools,
            })

        reward = GlobalEfficiencyReward(
            state_path=str(state_path),
            reference=str(ge_cfg.get("reference", "mean")),
            round_weight=float(ge_cfg.get("round_weight", 0.1) or 0.0),
            tool_weight=float(ge_cfg.get("tool_weight", 0.1) or 0.0),
            metric_tool_key=str(ge_cfg.get("metric_tool_key", "tool_calls_expanded")),
        )
        return reward.update_many(records)

    def _collect_efficiency_timing_metrics(self, timing_raw: dict, epoch: int) -> dict:
        rollout_s = float(timing_raw.get("generate_trajectories", 0.0) or 0.0)
        actor_forward_s = float(timing_raw.get("old_log_prob", 0.0) or 0.0)
        ref_forward_s = float(timing_raw.get("ref", 0.0) or 0.0)
        value_forward_s = float(timing_raw.get("values", 0.0) or 0.0)
        forward_s = actor_forward_s + ref_forward_s + value_forward_s
        actor_update_s = float(timing_raw.get("update_actor", 0.0) or 0.0)
        critic_update_s = float(timing_raw.get("update_critic", 0.0) or 0.0)
        update_s = actor_update_s + critic_update_s
        adv_s = float(timing_raw.get("adv", 0.0) or 0.0)
        train_only_s = rollout_s + forward_s + adv_s + update_s
        eval_s = float(timing_raw.get("testing", 0.0) or 0.0)
        save_checkpoint_s = float(timing_raw.get("save_checkpoint", 0.0) or 0.0)
        step_s = float(timing_raw.get("step", 0.0) or 0.0)

        record = {
            "step": self.global_steps,
            "epoch": epoch,
            "rollout_s": rollout_s,
            "actor_forward_s": actor_forward_s,
            "ref_forward_s": ref_forward_s,
            "value_forward_s": value_forward_s,
            "forward_s": forward_s,
            "actor_update_s": actor_update_s,
            "critic_update_s": critic_update_s,
            "update_s": update_s,
            "adv_s": adv_s,
            "train_only_s": train_only_s,
            "eval_s": eval_s,
            "save_checkpoint_s": save_checkpoint_s,
            "step_s": step_s,
            "rollout_fraction": rollout_s / train_only_s if train_only_s > 0 else 0.0,
        }

        log_path = os.path.join(
            self.config.trainer.default_local_dir, "efficiency_timing_log.jsonl"
        )
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return {
            f"efficiency/{key}": value
            for key, value in record.items()
            if key not in {"step", "epoch"}
        }

    def _get_mastered_state_checkpoint_folder(self):
        resume_mode = self.config.trainer.get("resume_mode", "auto")
        if resume_mode == "disable":
            return None

        if resume_mode == "resume_path":
            global_step_folder = self.config.trainer.get("resume_from_path", None)
            if global_step_folder is None:
                return None
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
            return global_step_folder

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
        return find_latest_ckpt_path(checkpoint_folder)

    def _load_checkpoint(self):
        global_step = super()._load_checkpoint()
        if not self.mastered_filter_enabled:
            return global_step
        if not self.mastered_filter_cfg.get("resume_with_mastered_state", True):
            print("Mastered sample filter: resume requested without restoring mastered state")
            return global_step
        global_step_folder = self._get_mastered_state_checkpoint_folder()
        if global_step_folder is None:
            return global_step
        state_path = os.path.join(global_step_folder, "mastered_sample_filter_state.json")
        if os.path.exists(state_path) and self.mastered_sampler is not None:
            with open(state_path, "r") as f:
                state = json.load(f)
            self.mastered_sampler.load_state_dict(state)
        else:
            print(f"Mastered sample filter: no saved state found at {state_path}, starting with empty blacklist")
        return global_step

    def fit_agent(self):
        """
        The training loop of PPO. Adapted to train the underlying model of agent.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        import time

        start_time = time.time()
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            self.agent_execution_engine.set_training_step(
                self.global_steps, mode="val", epoch=0,
                total_steps=self.total_training_steps
            )
            val_metrics = self._validate_agent()
            display_val_metrics = self._filter_eval_metrics_for_output(val_metrics)
            pprint(f"Initial validation metrics: {display_val_metrics}")
            logger.log(data=display_val_metrics, step=self.global_steps)
            self._save_eval_metrics(val_metrics, self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        print(f"Time taken to validate agent: {time.time() - start_time}")

        batch = None
        solve_none = 0
        solve_all = 0
        solve_partial = 0
        num_tasks = 0
        termination_counts = Counter()
        workflow_metrics = defaultdict(list)
        metrics = {}
        timing_raw = {}

        train_start_time = time.time()
        steps_completed_at_start = self.global_steps

        for epoch in range(self.config.trainer.total_epochs):
            if self.mastered_sampler is not None:
                self.mastered_sampler.set_epoch(epoch)
                active = self.mastered_sampler.num_active()
                mastered = self.mastered_sampler.num_mastered()
                total = self.mastered_sampler.total_size()
                pct = mastered * 100 // max(total, 1)
                pprint(
                    f"Epoch {epoch} | Step {self.global_steps}/{self.total_training_steps} | "
                    f"Samples: active={active} mastered={mastered} total={total} ({pct}% filtered)"
                )
                self._mastered_filter_log.append({
                    "event": "epoch_start", "epoch": epoch, "step": self.global_steps,
                    "active": active, "mastered": mastered, "total": total,
                })
            else:
                pprint(f"Epoch {epoch} | Step {self.global_steps}/{self.total_training_steps}")
            for batch_dict in self.train_dataloader:
                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.get("profile_steps") is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_tasks += len(new_batch.batch)

                source_task_ids = new_batch.non_tensor_batch.get("id", None)
                if source_task_ids is None:
                    task_ids = [str(uuid.uuid4()) for _ in range(len(new_batch.batch))]
                else:
                    task_ids = []
                    for sample_id in source_task_ids:
                        sid = "" if sample_id is None else str(sample_id)
                        task_ids.append(sid if sid else str(uuid.uuid4()))
                new_batch.non_tensor_batch["task_ids"] = np.array(task_ids, dtype=object)

                # Rollout sampling config
                dynamic_cfg = self.config.rllm.get("dynamic_rollout", None)
                dapo_cfg = self.config.rllm.get("dapo_sampling", None)
                selection_info = None
                use_dynamic = (
                    dynamic_cfg is not None
                    and dynamic_cfg.get("enable", False)
                    and not self.config.trainer.get("val_only", False)
                )
                use_dapo = (
                    dapo_cfg is not None
                    and dapo_cfg.get("enable", False)
                    and not self.config.trainer.get("val_only", False)
                )
                use_dapo_dynamic = use_dynamic and use_dapo
                use_full_rollouts_for_advantage = bool(
                    dynamic_cfg.get("use_full_rollouts_for_advantage", False)
                ) if use_dynamic else False
                full_rollout_advantage_stats = None
                mastered_full_group_correctness = None
                mastered_full_group_min_rollouts = None
                mastered_full_group_sample_ids = {}
                dapo_group_ids = None
                if use_dapo_dynamic:
                    min_rollouts = int(dynamic_cfg.min_rollouts)
                    max_rollouts = int(dynamic_cfg.max_rollouts)
                    if max_rollouts < min_rollouts:
                        raise ValueError("dynamic_rollout.max_rollouts must be >= min_rollouts")
                    dapo_group_ids = [str(uid) for uid in new_batch.non_tensor_batch["task_ids"]]
                    if "id" in new_batch.non_tensor_batch:
                        mastered_full_group_sample_ids = {
                            str(task_id): str(sample_id)
                            for task_id, sample_id in zip(
                                dapo_group_ids,
                                new_batch.non_tensor_batch["id"],
                                strict=False,
                            )
                        }
                    initial_batch = new_batch.repeat(
                        repeat_times=min_rollouts,
                        interleave=True,
                    )
                    if max_rollouts > min_rollouts:
                        extra_batch = new_batch.repeat(
                            repeat_times=max_rollouts - min_rollouts,
                            interleave=False,
                        )
                        new_batch = DataProto.concat([initial_batch, extra_batch])
                    else:
                        new_batch = initial_batch
                    n_rollouts = max_rollouts
                else:
                    if use_dynamic:
                        n_rollouts = dynamic_cfg.max_rollouts
                    else:
                        n_rollouts = self.config.actor_rollout_ref.rollout.n

                    new_batch = new_batch.repeat(
                        repeat_times=n_rollouts,
                        interleave=not use_dynamic,
                    )

                new_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

                # Update training step in engine for episode logging
                self.agent_execution_engine.set_training_step(
                    self.global_steps, mode="train", epoch=epoch,
                    total_steps=self.total_training_steps
                )

                with marked_timer("step", timing_raw):
                    # Build rollout sampling callbacks if enabled
                    if use_dapo_dynamic:
                        target_groups = dapo_cfg.get("target_batch_size", None)
                        target_groups = int(target_groups) if target_groups is not None else int(self.config.data.train_batch_size)
                        reward_std_threshold = float(dapo_cfg.get("reward_std_threshold", 1e-6))
                        dapo_signal_source = _get_dapo_signal_source(dapo_cfg)
                        completion_ratio_threshold = float(
                            dynamic_cfg.get("completion_ratio_threshold", 1.0)
                        )
                        selection_strategy = dynamic_cfg.get("selection_strategy", "first")
                        expansion_policy = _get_dynamic_expansion_policy(dynamic_cfg)
                        completed_rewards = defaultdict(list)
                        completed_correctness = defaultdict(list)
                        completed_group_ids = set()
                        valid_group_ids = []
                        invalid_group_ids = []
                        dapo_dynamic_stop_summary = {}

                        def _completed_count(task_id, task_states):
                            state = task_states.get(task_id)
                            return int(state["completed"]) if state is not None else 0

                        def _raw_group_has_signal(task_id, completed_count):
                            if completed_count < min_rollouts:
                                return False
                            return _dapo_group_has_signal(
                                completed_rewards.get(task_id, []), reward_std_threshold,
                                dapo_signal_source,
                            )

                        def _raw_group_has_any_correct(task_id, completed_count):
                            if completed_count < min_rollouts:
                                return False
                            return any(
                                bool(value)
                                for value in completed_correctness.get(task_id, [])
                                if value is not None
                            )

                        def on_episode_done(task_id, episode):
                            completed_rewards[task_id].append(
                                _get_episode_dapo_signal(episode, dapo_signal_source)
                            )
                            completed_correctness[task_id].append(
                                _get_episode_correctness(episode)
                            )

                        def schedule_condition(task_id, rollout_idx, state):
                            completed_count = int(state["completed"])
                            if rollout_idx < min_rollouts:
                                return True
                            if completed_count < min_rollouts:
                                return True
                            if expansion_policy == "expand_only_all_wrong":
                                return not _raw_group_has_any_correct(task_id, completed_count)
                            return not _raw_group_has_signal(task_id, completed_count)

                        def stop_condition(task_states):
                            completed_min_ids = [
                                task_id for task_id in dapo_group_ids
                                if _completed_count(task_id, task_states) >= min_rollouts
                            ]
                            raw_valid_ids = [
                                task_id for task_id in dapo_group_ids
                                if _raw_group_has_signal(task_id, _completed_count(task_id, task_states))
                            ]
                            raw_invalid_ids = [
                                task_id for task_id in completed_min_ids
                                if task_id not in raw_valid_ids
                            ]
                            completed_group_ids.clear()
                            completed_group_ids.update(completed_min_ids)
                            valid_group_ids[:] = raw_valid_ids
                            invalid_group_ids[:] = raw_invalid_ids
                            total_completed = sum(
                                int(state["completed"]) for state in task_states.values()
                            )
                            total_skipped = sum(
                                int(state.get("skipped", 0)) for state in task_states.values()
                            )
                            total_candidates = max(len(dapo_group_ids) * max_rollouts, 1)
                            completion_ratio = (
                                (total_completed + total_skipped) / total_candidates
                            )
                            stop_summary = {
                                "target_groups": int(target_groups),
                                "valid_groups": int(len(raw_valid_ids)),
                                "raw_groups": int(len(dapo_group_ids)),
                                "min_completed_groups": int(len(completed_min_ids)),
                                "completed_rollouts": int(total_completed),
                                "skipped_rollouts": int(total_skipped),
                                "total_candidate_rollouts": int(total_candidates),
                                "completion_ratio": float(completion_ratio),
                                "completion_ratio_threshold": float(completion_ratio_threshold),
                            }
                            dapo_dynamic_stop_summary.clear()
                            dapo_dynamic_stop_summary.update(stop_summary)
                            if len(raw_valid_ids) >= target_groups:
                                dapo_dynamic_stop_summary["reason"] = "target_valid_groups_met"
                                return (
                                    f"dapo_dynamic_valid_groups={len(raw_valid_ids)}/{target_groups} "
                                    f"completion_ratio={completion_ratio:.3f}/{completion_ratio_threshold:.3f}"
                                )
                            if completion_ratio >= completion_ratio_threshold:
                                dapo_dynamic_stop_summary["reason"] = "fallback_total_completion"
                                return (
                                    f"completion_ratio={completion_ratio:.3f}/{completion_ratio_threshold:.3f} "
                                    f"valid_groups={len(raw_valid_ids)}/{target_groups} "
                                    f"accounted_rollouts={total_completed + total_skipped}/{total_candidates}"
                                )
                            return ""

                        final_gen_batch_output, batch_task_ids = self.generate_trajectories(
                            batch=new_batch, timing_raw=timing_raw,
                            stop_condition=stop_condition,
                            on_episode_done=on_episode_done,
                            min_rollouts=min_rollouts,
                            sampling_mode="DAPO+dynamic rollout",
                            schedule_condition=schedule_condition,
                        )

                        new_batch = _merge_dapo_generated_groups(
                            final_gen_batch_output, new_batch,
                        )
                        mastered_full_group_correctness = completed_correctness
                        mastered_full_group_min_rollouts = min_rollouts
                        if use_full_rollouts_for_advantage:
                            full_rollout_advantage_stats = _compute_group_reward_stats(new_batch)
                        final_selection_strategy = dapo_cfg.get("final_selection_strategy", "first")
                        new_batch, selection_info = _filter_dapo_groups_after_reward(
                            new_batch, target_groups,
                            reward_std_threshold, final_selection_strategy,
                            dapo_signal_source,
                            min_rollouts,
                        )
                        dapo_group_selection_info = dict(selection_info)
                        new_batch, n_rollouts, rollout_selection_info = _select_batch_rollouts_per_group(
                            new_batch, min_rollouts, selection_strategy,
                        )
                        selection_info["selection_strategy"] = f"dapo_dynamic_group_filter_after_{dapo_signal_source}_signal"
                        selection_info["min_rollouts"] = int(min_rollouts)
                        selection_info["max_rollouts"] = int(max_rollouts)
                        selection_info["n_rollouts"] = int(n_rollouts)
                        selection_info["dynamic_expansion_policy"] = expansion_policy
                        selection_info["completion_ratio_threshold"] = float(completion_ratio_threshold)
                        selection_info["completion_ratio_scope"] = "accounted_total_candidates"
                        selection_info["dapo_dynamic_stop_summary"] = dict(dapo_dynamic_stop_summary)
                        selection_info["dapo_dynamic_stop_reason"] = dapo_dynamic_stop_summary.get("reason", "natural_exhaustion")
                        selection_info["completed_raw_group_ids"] = list(completed_group_ids)
                        selection_info["completed_raw_valid_group_ids"] = list(valid_group_ids)
                        selection_info["completed_raw_invalid_group_ids"] = list(invalid_group_ids)
                        selection_info["completed_raw_any_correct_group_ids"] = [
                            task_id for task_id in dapo_group_ids
                            if any(
                                bool(value)
                                for value in completed_correctness.get(task_id, [])
                                if value is not None
                            )
                        ]
                        selection_info["completed_raw_all_wrong_group_ids"] = [
                            task_id for task_id in completed_group_ids
                            if not any(
                                bool(value)
                                for value in completed_correctness.get(task_id, [])
                                if value is not None
                            )
                        ]
                        selection_info["dapo_group_selection"] = dapo_group_selection_info
                        selection_info["rollout_selection"] = rollout_selection_info
                        combined_per_sample = dict(dapo_group_selection_info.get("per_sample", {}) or {})
                        for group_key, rollout_sample_info in (rollout_selection_info.get("per_sample", {}) or {}).items():
                            dapo_sample_info = dict(combined_per_sample.get(group_key, {}) or {})
                            merged_sample_info = dict(dapo_sample_info)
                            merged_sample_info.update(rollout_sample_info)
                            merged_sample_info["dapo_selected_group"] = dapo_sample_info.get("selected_group", 0)
                            merged_sample_info["dapo_valid_signal"] = dapo_sample_info.get("valid_signal", 0)
                            combined_per_sample[group_key] = merged_sample_info
                        selection_info["per_sample"] = combined_per_sample
                        selection_info["selected_episode_ids"] = rollout_selection_info["selected_episode_ids"]
                        selection_info["discarded_episode_ids"] = sorted(
                            set(dapo_group_selection_info.get("discarded_episode_ids", []))
                            | set(rollout_selection_info.get("discarded_episode_ids", []))
                        )
                        rollout_save_dir = self.workflow_args.get("rollout_save_dir", None)
                        if rollout_save_dir is not None:
                            import json as _json
                            mode = "val" if getattr(self, "_is_validation", False) else "train"
                            manifest_dir = os.path.join(rollout_save_dir, mode, f"step_{self.global_steps}")
                            os.makedirs(manifest_dir, exist_ok=True)
                            manifest_path = os.path.join(manifest_dir, "selection_manifest.json")
                            with open(manifest_path, "w") as f:
                                _json.dump(selection_info, f, indent=2)
                    elif use_dynamic:
                        min_rollouts = dynamic_cfg.min_rollouts
                        signal_threshold = dynamic_cfg.signal_threshold
                        completion_ratio_threshold = dynamic_cfg.get("completion_ratio_threshold", 1.0)
                        selection_strategy = dynamic_cfg.get("selection_strategy", "diverse")
                        expansion_policy = _get_dynamic_expansion_policy(dynamic_cfg)
                        completed_rewards = defaultdict(list)
                        completed_correctness = defaultdict(list)

                        def on_episode_done(task_id, episode):
                            reward = _get_episode_reward(episode)
                            completed_rewards[task_id].append(reward)
                            completed_correctness[task_id].append(
                                _get_episode_correctness(episode)
                            )

                        def schedule_condition(task_id, rollout_idx, state):
                            if expansion_policy == "until_signal":
                                return True
                            completed_count = int(state["completed"])
                            if rollout_idx < min_rollouts:
                                return True
                            if completed_count < min_rollouts:
                                return True
                            if expansion_policy == "expand_only_all_wrong":
                                return not any(
                                    bool(value)
                                    for value in completed_correctness.get(task_id, [])
                                    if value is not None
                                )
                            return True

                        def stop_condition(task_states):
                            return _should_stop_dynamic_rollout(
                                task_states, completed_rewards,
                                min_rollouts, signal_threshold,
                                completion_ratio_threshold,
                            )

                        final_gen_batch_output, batch_task_ids = self.generate_trajectories(
                            batch=new_batch, timing_raw=timing_raw,
                            stop_condition=stop_condition,
                            on_episode_done=on_episode_done,
                            min_rollouts=min_rollouts,
                            sampling_mode="Dynamic rollout",
                            schedule_condition=(
                                schedule_condition
                                if expansion_policy != "until_signal"
                                else None
                            ),
                        )

                        if use_full_rollouts_for_advantage:
                            full_rollout_advantage_stats = _compute_group_reward_stats(final_gen_batch_output)
                        final_gen_batch_output, new_batch, n_rollouts, selection_info = _select_and_truncate(
                            final_gen_batch_output, new_batch,
                            min_rollouts, selection_strategy,
                        )
                        selection_info["dynamic_expansion_policy"] = expansion_policy
                        selection_info["completed_raw_any_correct_group_ids"] = [
                            task_id for task_id, values in completed_correctness.items()
                            if any(bool(value) for value in values if value is not None)
                        ]
                        selection_info["completed_raw_all_wrong_group_ids"] = [
                            task_id for task_id, values in completed_correctness.items()
                            if values and not any(bool(value) for value in values if value is not None)
                        ]
                        rollout_save_dir = self.workflow_args.get("rollout_save_dir", None)
                        if rollout_save_dir is not None:
                            import json as _json
                            mode = "val" if getattr(self, "_is_validation", False) else "train"
                            manifest_dir = os.path.join(rollout_save_dir, mode, f"step_{self.global_steps}")
                            os.makedirs(manifest_dir, exist_ok=True)
                            manifest_path = os.path.join(manifest_dir, "selection_manifest.json")
                            with open(manifest_path, "w") as f:
                                _json.dump(selection_info, f, indent=2)
                    elif use_dapo:
                        target_groups = dapo_cfg.get("target_batch_size", None)
                        target_groups = int(target_groups) if target_groups is not None else int(self.config.data.train_batch_size)
                        reward_std_threshold = float(dapo_cfg.get("reward_std_threshold", 1e-6))
                        dapo_signal_source = _get_dapo_signal_source(dapo_cfg)
                        completed_rewards = defaultdict(list)
                        completed_group_ids = set()
                        valid_group_ids = []
                        invalid_group_ids = []

                        def on_episode_done(task_id, episode):
                            completed_rewards[task_id].append(
                                _get_episode_dapo_signal(episode, dapo_signal_source)
                            )

                        def stop_condition(task_states):
                            for task_id, state in task_states.items():
                                if task_id in completed_group_ids:
                                    continue
                                if state["total_rollouts"] > 0 and state["completed"] >= state["total_rollouts"]:
                                    completed_group_ids.add(task_id)
                                    rewards = completed_rewards.get(task_id, [])
                                    if _dapo_group_has_signal(rewards, reward_std_threshold, dapo_signal_source):
                                        valid_group_ids.append(task_id)
                                    else:
                                        invalid_group_ids.append(task_id)
                            if len(valid_group_ids) >= target_groups:
                                return f"dapo_valid_groups={len(valid_group_ids)}/{target_groups}"
                            return ""

                        final_gen_batch_output, batch_task_ids = self.generate_trajectories(
                            batch=new_batch, timing_raw=timing_raw,
                            stop_condition=stop_condition,
                            on_episode_done=on_episode_done,
                            min_rollouts=n_rollouts,
                            sampling_mode="DAPO sampling",
                        )

                        new_batch = _merge_dapo_generated_groups(
                            final_gen_batch_output, new_batch,
                        )
                        final_selection_strategy = dapo_cfg.get("final_selection_strategy", "first")
                        new_batch, selection_info = _filter_dapo_groups_after_reward(
                            new_batch, target_groups,
                            reward_std_threshold, final_selection_strategy,
                            dapo_signal_source,
                            n_rollouts,
                        )
                        selection_info["completed_raw_valid_group_ids"] = list(valid_group_ids)
                        selection_info["completed_raw_invalid_group_ids"] = list(invalid_group_ids)
                        rollout_save_dir = self.workflow_args.get("rollout_save_dir", None)
                        if rollout_save_dir is not None:
                            import json as _json
                            mode = "val" if getattr(self, "_is_validation", False) else "train"
                            manifest_dir = os.path.join(rollout_save_dir, mode, f"step_{self.global_steps}")
                            os.makedirs(manifest_dir, exist_ok=True)
                            manifest_path = os.path.join(manifest_dir, "selection_manifest.json")
                            with open(manifest_path, "w") as f:
                                _json.dump(selection_info, f, indent=2)
                    else:
                        # generate trajectories (original path)
                        final_gen_batch_output = self.generate_trajectories(
                            batch=new_batch, timing_raw=timing_raw
                        )

                        # need to repeat to make shape match
                        repeat_counts = final_gen_batch_output.meta_info["repeat_counts"]
                        new_batch = new_batch.sample_level_repeat(repeat_counts)
                        final_gen_batch_output.meta_info.pop(
                            "repeat_counts", None
                        )  # no longer needed after this
                        new_batch = new_batch.union(final_gen_batch_output)

                    # rejection sampling
                    # we do rejection sampling at the episode level instead of the traj/step level
                    uids = new_batch.non_tensor_batch["task_ids"]
                    unique_uids = np.unique(uids)
                    is_correct = new_batch.non_tensor_batch["is_correct"]
                    drop_uids = set()

                    for uid in unique_uids:
                        candidate_rows = uids == uid
                        candidate_is_correct = is_correct[candidate_rows]

                        # Check if all episodes are correct or incorrect
                        if not candidate_is_correct.any():
                            drop_uids.add(uid)
                            solve_none += 1
                        elif candidate_is_correct.all():
                            drop_uids.add(uid)
                            solve_all += 1
                        else:
                            solve_partial += 1

                    # Build a view with a single item per episode_id for metrics/logging
                    seen_episodes = set()
                    episode_unique_idxs = []
                    for i, episode_id in enumerate(
                        new_batch.non_tensor_batch["episode_ids"]
                    ):
                        if episode_id not in seen_episodes:
                            seen_episodes.add(episode_id)
                            episode_unique_idxs.append(i)
                    episode_unique_batch = new_batch.select_idxs(episode_unique_idxs)

                    # log metrics returned by workflows
                    for metric_dict in episode_unique_batch.non_tensor_batch["metrics"]:
                        for key, value in metric_dict.items():
                            metric_value = _to_finite_float(value)
                            if metric_value is not None:
                                workflow_metrics[key].append(metric_value)
                    metrics.update(
                        self._collect_rollout_overview_metrics(episode_unique_batch, selection_info)
                    )

                    # collect and log termination reasons
                    termination_reasons = episode_unique_batch.non_tensor_batch[
                        "termination_reasons"
                    ]
                    termination_counts.update(termination_reasons)

                    # If no valid samples remain, skip this batch and get a new one
                    # if len(drop_uids) == len(unique_uids):
                    #     print("No valid samples remain, skipping batch")
                    #     continue

                    if use_dapo:
                        sampling_label = "DAPO+dynamic rollout" if use_dapo_dynamic else "DAPO sampling"
                        if new_batch.batch["input_ids"].shape[0] == 0:
                            colorful_print(
                                f"{sampling_label}: no valid reward-signal groups after reward shaping; skipping this batch",
                                fg="yellow",
                            )
                            continue

                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        target_groups = dapo_cfg.get("target_batch_size", None)
                        target_groups = int(target_groups) if target_groups is not None else int(self.config.data.train_batch_size)
                        unique_uids = list(dict.fromkeys(str(uid) for uid in batch.non_tensor_batch["task_ids"]))
                        if len(unique_uids) < target_groups:
                            colorful_print(
                                f"{sampling_label}: produced {len(unique_uids)}/{target_groups} shaped-valid groups; updating with available groups",
                                fg="yellow",
                            )
                    elif not self.config.rllm.rejection_sample.enable:
                        batch = new_batch
                    else:
                        rejection_mask = np.isin(uids, list(drop_uids))
                        new_batch = new_batch[~rejection_mask]
                        if batch is None:
                            batch = new_batch
                        else:
                            batch = DataProto.concat([batch, new_batch])

                        if solve_partial < self.config.data.train_batch_size:
                            continue
                        else:
                            # randomly select bsz task uids from batch, then filter batch to only contain these tasks
                            # TODO: add heuristic for selecting train_batch_size uids
                            uids = batch.non_tensor_batch["task_ids"]
                            unique_uids = np.unique(uids)
                            assert (
                                len(unique_uids) >= self.config.data.train_batch_size
                            ), "Not enough unique uids to sample from"
                            selected_uids = np.random.choice(
                                unique_uids,
                                size=self.config.data.train_batch_size,
                                replace=False,
                            )
                            selected_mask = np.isin(uids, selected_uids)
                            batch = batch[selected_mask]

                    if full_rollout_advantage_stats:
                        batch = _attach_full_rollout_advantage_stats(
                            batch, full_rollout_advantage_stats,
                        )
                        metrics["rollout_sampling/full_rollout_advantage_groups"] = float(
                            len(full_rollout_advantage_stats)
                        )

                    # Update global efficiency state once per step, after rollout
                    # filtering/selection has decided which episodes this batch keeps.
                    seen_update_episodes = set()
                    update_episode_unique_idxs = []
                    for i, episode_id in enumerate(batch.non_tensor_batch["episode_ids"]):
                        if episode_id not in seen_update_episodes:
                            seen_update_episodes.add(episode_id)
                            update_episode_unique_idxs.append(i)
                    if update_episode_unique_idxs:
                        global_efficiency_state_updates = self._update_global_efficiency_state_after_step(
                            batch.select_idxs(update_episode_unique_idxs)
                        )
                        if global_efficiency_state_updates:
                            metrics["rollout_overview/global_efficiency_state_updates"] = float(
                                global_efficiency_state_updates
                            )

                    if (
                        self.config.rllm.stepwise_advantage.enable
                        and self.config.rllm.stepwise_advantage.mode == "broadcast"
                    ):
                        # need to make sure both number of last steps (number of uids) and number of total steps in the batch
                        # (batch size after processing) are both multiples of world size

                        # first we split the batch in two: one with only the last steps of each trajectory and the other with the remaining steps
                        is_last_step = batch.non_tensor_batch["is_last_step"]
                        valid_last_step_indices = np.where(is_last_step == True)[0]
                        not_last_step_indices = np.where(is_last_step == False)[0]
                        last_step_batch = batch.select_idxs(
                            valid_last_step_indices
                        )  # This batch only has valid last steps
                        non_last_step_batch = batch.select_idxs(not_last_step_indices)

                        # round down last_step_batch to make sure its multiple of world size
                        num_trainer_replicas = self.actor_rollout_wg.world_size
                        max_batch_size = (
                            last_step_batch.batch["input_ids"].shape[0]
                            // num_trainer_replicas
                        ) * num_trainer_replicas

                        size_mask = torch.zeros(
                            last_step_batch.batch["input_ids"].shape[0],
                            dtype=torch.bool,
                        )
                        size_mask[:max_batch_size] = True
                        last_step_batch = last_step_batch[
                            size_mask
                        ]  # filtered last steps

                        # Keep only non-last steps whose exact rollout still has a
                        # retained last step. Multiple rollout episode_ids can share
                        # the same trajectory_id, so filtering by trajectory_id alone
                        # leaves orphaned non-last steps after world-size rounding.
                        valid_last_step_pairs = set(
                            zip(
                                last_step_batch.non_tensor_batch["trajectory_ids"],
                                last_step_batch.non_tensor_batch["episode_ids"],
                                strict=False,
                            )
                        )
                        non_last_step_pairs = zip(
                            non_last_step_batch.non_tensor_batch["trajectory_ids"],
                            non_last_step_batch.non_tensor_batch["episode_ids"],
                            strict=False,
                        )
                        non_last_step_mask = np.array(
                            [pair in valid_last_step_pairs for pair in non_last_step_pairs],
                            dtype=bool,
                        )
                        non_last_step_batch = non_last_step_batch[non_last_step_mask]

                        # concatenate then pad
                        batch = DataProto.concat([last_step_batch, non_last_step_batch])
                        batch = self._pad_dataproto_to_world_size(batch)

                    else:
                        # then we just pad the batch size to a multiple of world size
                        batch = self._pad_dataproto_to_world_size(batch=batch)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = (
                            self.config.actor_rollout_ref.actor.loss_agg_mode
                        )
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=loss_agg_mode,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item()
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(
                                rollout_probs_diff, response_mask.bool()
                            )
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                    batch
                                )
                            else:
                                ref_log_prob = (
                                    self.actor_rollout_wg.compute_ref_log_prob(batch)
                                )
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # Broadcast computes the final-step advantage across rollouts
                        # of the same prompt, then applies it to every step in each
                        # rollout. Use trajectory_ids here; step_ids include the step
                        # index and would split rollouts with different final lengths
                        # into separate RLOO groups.
                        if (
                            self.config.rllm.stepwise_advantage.enable
                            and self.config.rllm.stepwise_advantage.mode == "broadcast"
                        ):
                            batch.non_tensor_batch["uid"] = batch.non_tensor_batch[
                                "trajectory_ids"
                            ]
                        else:
                            batch.non_tensor_batch["uid"] = batch.non_tensor_batch[
                                "step_ids"
                            ]

                        if (
                            self.config.rllm.stepwise_advantage.enable
                            and self.config.rllm.stepwise_advantage.mode == "per_step"
                        ):
                            batch.batch["token_level_scores"] = batch.batch[
                                "step_rewards"
                            ]
                        else:
                            batch.batch["token_level_scores"] = batch.batch[
                                "traj_rewards"
                            ]

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch[
                                "token_level_scores"
                            ]

                        if (
                            self.config.rllm.stepwise_advantage.enable
                            and self.config.rllm.stepwise_advantage.mode == "broadcast"
                        ):
                            is_last_step = batch.non_tensor_batch["is_last_step"]
                            last_step_indices = np.where(is_last_step == True)[0]
                            not_last_step_indices = np.where(is_last_step == False)[0]
                            non_last_step_batch = batch.select_idxs(
                                not_last_step_indices
                            )
                            batch = batch.select_idxs(
                                last_step_indices
                            )  # This batch only has last steps
                            # last_step_batch contains no padded steps as it was rounded down (not padded) to a multiple of world size
                        else:
                            batch = self._remove_padding(
                                batch
                            )  # compute advantages over non-padded steps only

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=n_rollouts,
                            norm_adv_by_std_in_grpo=self.config.algorithm.norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        if (
                            "full_rollout_advantage_mean" in batch.non_tensor_batch
                            and self.config.algorithm.adv_estimator in (
                                AdvantageEstimator.RLOO,
                                AdvantageEstimator.RLOO_VECTORIZED,
                            )
                        ):
                            batch = _apply_full_rollout_rloo_advantage(batch)
                            metrics["rollout_sampling/full_rollout_advantage_enabled"] = 1.0

                        if (
                            self.config.rllm.stepwise_advantage.enable
                            and self.config.rllm.stepwise_advantage.mode == "broadcast"
                            and len(non_last_step_batch) > 0
                        ):
                            # Merging the separated out steps using the advantage from last steps
                            self._stepwise_advantage_broadcast(
                                batch, non_last_step_batch
                            )
                            batch = DataProto.concat([batch, non_last_step_batch])

                    # remove invalid items filtered out due to compact filtering
                    is_valid = batch.non_tensor_batch["is_valid"]
                    valid_idxs = np.where(is_valid == True)[0]
                    batch = batch.select_idxs(valid_idxs)

                    # for backward compatibility
                    if self.config.rllm.mask_truncated_samples:
                        mask = batch.batch["attention_mask"][:, -1] == 1
                        batch = batch[~mask]

                    # re-pad batch size to world size for gradient update
                    batch = self._pad_dataproto_to_world_size(batch=batch)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and self.global_steps > 0
                        and self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            self.agent_execution_engine.set_training_step(
                                self.global_steps, mode="val", epoch=epoch,
                                total_steps=self.total_training_steps
                            )
                            val_metrics: dict = self._validate_agent()
                        metrics.update(self._filter_eval_metrics_for_output(val_metrics))
                        # Save eval metrics to disk
                        self._save_eval_metrics(val_metrics, self.global_steps)

                    if (
                        self.config.trainer.save_freq > 0
                        and self.global_steps > 0
                        and self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    # # Visualize some sample trajectories
                    # if batch is not None and len(batch) > 0:
                    #     # Randomly select a few samples to visualize
                    #     batch_size = len(batch)
                    #     num_samples = min(2, batch_size)  # Visualize up to 2 samples
                    #     if num_samples > 0:
                    #         sample_indices = np.random.choice(
                    #             batch_size, size=num_samples, replace=False
                    #         )
                    #         for idx in sample_indices:
                    #             self.visualize_trajectory_last_step(
                    #                 batch, sample_idx=idx, max_samples=1
                    #             )

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                metrics.update(
                    self._collect_efficiency_timing_metrics(timing_raw, epoch)
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                metrics["batch/solve_none"] = solve_none / num_tasks
                metrics["batch/solve_all"] = solve_all / num_tasks
                metrics["batch/solve_partial"] = solve_partial / num_tasks

                for key, value in workflow_metrics.items():
                    metrics[f"batch/{key}"] = np.mean(value)

                for r in TerminationReason:
                    metrics[f"batch/{r.value}"] = termination_counts[r.value] / len(
                        set(new_batch.non_tensor_batch["episode_ids"])
                    )

                metrics["batch/num_tasks"] = num_tasks

                early_stop_due_to_mastered_filter = False

                if self.mastered_filter_enabled and self.mastered_sampler is not None:
                    id_key = None
                    sample_to_correct = defaultdict(list)
                    mastered_filter_source = "selected_batch"
                    mastered_filter_min_rollouts = 1
                    insufficient_rollouts = 0

                    if mastered_full_group_correctness is not None:
                        mastered_filter_source = "completed_rollouts"
                        mastered_filter_min_rollouts = int(mastered_full_group_min_rollouts or 1)
                        for task_id, values in mastered_full_group_correctness.items():
                            sample_id = mastered_full_group_sample_ids.get(str(task_id), str(task_id))
                            sample_to_correct[str(sample_id)].extend(
                                bool(value) for value in values if value is not None
                            )
                    else:
                        if "id" in episode_unique_batch.non_tensor_batch:
                            id_key = "id"
                        else:
                            if not self._mastered_filter_id_warning_logged:
                                colorful_print(
                                    f"WARNING: mastered_sample_filter enabled but 'id' not found in batch. "
                                    f"Available keys: {list(episode_unique_batch.non_tensor_batch.keys())}",
                                    fg="red",
                                )
                                self._mastered_filter_id_warning_logged = True

                        if id_key is not None:
                            step_sample_ids = np.array(
                                [str(x) for x in episode_unique_batch.non_tensor_batch[id_key]],
                                dtype=object,
                            )
                            step_is_correct = episode_unique_batch.non_tensor_batch["is_correct"]
                            for sample_id, ic in zip(step_sample_ids, step_is_correct, strict=False):
                                sample_to_correct[str(sample_id)].append(bool(ic))

                    if sample_to_correct:
                        newly_mastered = [
                            sample_id for sample_id, vals in sample_to_correct.items()
                            if len(vals) >= mastered_filter_min_rollouts and all(vals)
                        ]
                        added_mastered = self.mastered_sampler.mark_mastered(newly_mastered) if newly_mastered else 0
                        metrics["mastered_filter/newly_mastered"] = added_mastered
                        metrics["mastered_filter/num_mastered"] = self.mastered_sampler.num_mastered()
                        metrics["mastered_filter/num_active"] = self.mastered_sampler.num_active()
                        metrics["mastered_filter/epoch"] = epoch

                        eligible_groups = {
                            sample_id: vals
                            for sample_id, vals in sample_to_correct.items()
                            if len(vals) >= mastered_filter_min_rollouts
                        }
                        insufficient_rollouts = len(sample_to_correct) - len(eligible_groups)
                        fully_correct = sum(1 for vals in eligible_groups.values() if vals and all(vals))
                        partial = sum(1 for vals in eligible_groups.values() if vals and any(vals) and not all(vals))
                        all_wrong = sum(1 for vals in eligible_groups.values() if vals and not any(vals))

                        step_record = {
                            "event": "step", "epoch": epoch,
                            "step": self.global_steps + 1,
                            "total_steps": self.total_training_steps,
                            "newly_mastered": added_mastered,
                            "newly_mastered_ids": newly_mastered if added_mastered > 0 else [],
                            "cumulative_mastered": self.mastered_sampler.num_mastered(),
                            "active": self.mastered_sampler.num_active(),
                            "total": self.mastered_sampler.total_size(),
                            "source": mastered_filter_source,
                            "min_rollouts": mastered_filter_min_rollouts,
                            "batch_fully_correct": fully_correct,
                            "batch_partial": partial,
                            "batch_all_wrong": all_wrong,
                            "batch_insufficient_rollouts": insufficient_rollouts,
                            "batch_total_samples": len(sample_to_correct),
                        }
                        self._mastered_filter_log.append(step_record)

                        log_freq = int(self.mastered_filter_cfg.get("log_freq", 1) or 1)
                        if log_freq > 0 and ((self.global_steps + 1) % log_freq == 0):
                            pct = self.mastered_sampler.num_mastered() * 100 // max(self.mastered_sampler.total_size(), 1)
                            colorful_print(
                                f"Mastered filter | epoch={epoch} step={self.global_steps + 1}/{self.total_training_steps} "
                                f"source={mastered_filter_source} min_rollouts={mastered_filter_min_rollouts} "
                                f"newly={added_mastered} mastered={self.mastered_sampler.num_mastered()} "
                                f"active={self.mastered_sampler.num_active()} total={self.mastered_sampler.total_size()} "
                                f"({pct}% filtered) | "
                                f"batch: correct={fully_correct} partial={partial} wrong={all_wrong} "
                                f"insufficient={insufficient_rollouts}",
                                fg="red",
                            )

                        if self.mastered_filter_cfg.get("early_stop_when_insufficient", True):
                            threshold = self.mastered_filter_cfg.get("min_active_samples", None)
                            if threshold is None:
                                threshold = self.config.data.train_batch_size
                            if self.mastered_sampler.num_active() < int(threshold):
                                colorful_print(
                                    f"Early stop: mastered blacklist left insufficient active samples | "
                                    f"total={self.mastered_sampler.total_size()} mastered={self.mastered_sampler.num_mastered()} "
                                    f"active={self.mastered_sampler.num_active()} threshold={int(threshold)}",
                                    fg="red",
                                )
                                early_stop_due_to_mastered_filter = True

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if early_stop_due_to_mastered_filter:
                    self._save_checkpoint()
                    return

                self.global_steps += 1

                # Print step timing summary with ETA
                elapsed = time.time() - train_start_time
                done = self.global_steps - steps_completed_at_start
                remaining = self.total_training_steps - self.global_steps
                avg_per_step = elapsed / done if done > 0 else 0
                eta = avg_per_step * remaining

                def _fmt_time(seconds):
                    h, r = divmod(int(seconds), 3600)
                    m, s = divmod(r, 60)
                    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

                pprint(
                    f"Step {self.global_steps}/{self.total_training_steps} done | "
                    f"Elapsed: {_fmt_time(elapsed)} | ETA: {_fmt_time(eta)} | "
                    f"Avg: {_fmt_time(avg_per_step)}/step"
                )

                batch = None
                solve_none = 0
                solve_all = 0
                solve_partial = 0
                num_tasks = 0
                termination_counts = Counter()
                workflow_metrics = defaultdict(list)
                metrics = {}
                timing_raw = {}


                if self.global_steps >= self.total_training_steps:
                    # Always save checkpoint at the final step
                    self._save_checkpoint()

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        self.agent_execution_engine.set_training_step(
                            self.global_steps, mode="val", epoch=epoch,
                            total_steps=self.total_training_steps
                        )
                        val_metrics = self._validate_agent()
                        display_val_metrics = self._filter_eval_metrics_for_output(val_metrics)
                        pprint(f"Final validation metrics: {display_val_metrics}")
                        logger.log(data=display_val_metrics, step=self.global_steps)
                        self._save_eval_metrics(val_metrics, self.global_steps)
                    return

    @staticmethod
    def _filter_eval_metrics_for_output(val_metrics: dict) -> dict:
        """Keep eval display and saved files focused on headline metrics."""
        return {
            k: v
            for k, v in val_metrics.items()
            if k.endswith("/pass@1") or k.endswith("/completed")
        }

    def _save_eval_metrics(self, val_metrics: dict, step: int):
        """Save evaluation metrics to a JSON file in the output directory."""
        import json
        import os

        eval_dir = os.path.join(
            self.config.trainer.get("default_local_dir", "checkpoints"),
            "eval_results",
        )
        os.makedirs(eval_dir, exist_ok=True)

        eval_file = os.path.join(eval_dir, f"eval_step_{step}.json")
        # Convert numpy types to native Python for JSON serialization.
        serializable_metrics = {
            k: float(v) if hasattr(v, "item") else v
            for k, v in self._filter_eval_metrics_for_output(val_metrics).items()
        }
        serializable_metrics["step"] = step

        with open(eval_file, "w") as f:
            json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)

        # Also append to a summary file for easy tracking
        summary_file = os.path.join(eval_dir, "eval_summary.jsonl")
        with open(summary_file, "a") as f:
            f.write(json.dumps(serializable_metrics, ensure_ascii=False) + "\n")

        print(f"[Eval] Saved eval metrics to {eval_file}")

    def _validate_agent(self):
        is_correct_lst = []
        data_source_lst = []
        uid_lst = []
        workflow_metrics_by_source = defaultdict(lambda: defaultdict(list))

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            source_task_ids = test_batch.non_tensor_batch.get("id", None)
            if source_task_ids is None:
                task_ids = [str(uuid.uuid4()) for _ in range(len(test_batch.batch))]
            else:
                task_ids = []
                for sample_id in source_task_ids:
                    sid = "" if sample_id is None else str(sample_id)
                    task_ids.append(sid if sid else str(uuid.uuid4()))
            test_batch.non_tensor_batch["task_ids"] = np.array(task_ids, dtype=object)

            n_val_samples = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)

            test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids"],
            )  # these are not needed for environment based interaction
            test_batch.meta_info = {"validate": True}

            test_output_gen_batch = self.generate_trajectories(batch=test_batch)
            repeat_counts = test_output_gen_batch.meta_info["repeat_counts"]
            # need to repeat to make shape match
            test_batch = test_batch.sample_level_repeat(repeat_counts)
            test_output_gen_batch.meta_info.pop(
                "repeat_counts", None
            )  # no longer needed after this
            test_batch = test_batch.union(test_output_gen_batch)

            seen_episodes = set()
            selected_idxs = []
            for i, episode_id in enumerate(test_batch.non_tensor_batch["episode_ids"]):
                if episode_id not in seen_episodes:
                    seen_episodes.add(episode_id)
                    selected_idxs.append(i)
            test_batch = test_batch.select_idxs(selected_idxs)

            is_correct_lst.extend(test_batch.non_tensor_batch["is_correct"])
            uid_lst.extend(test_batch.non_tensor_batch["task_ids"])

            data_sources = test_batch.non_tensor_batch.get("data_source", None)
            if data_sources is None:
                data_sources = ["unknown"] * len(test_batch)
            data_source_lst.extend(data_sources)

            # Collect workflow metrics per episode and data source
            for i, data_source in enumerate(data_sources):
                episode_metrics = test_batch.non_tensor_batch["metrics"][i]
                if episode_metrics is not None:
                    for key, value in episode_metrics.items():
                        metric_value = _to_finite_float(value)
                        if metric_value is not None:
                            workflow_metrics_by_source[data_source][key].append(metric_value)

        metrics = {}
        is_correct_array = np.array(is_correct_lst)
        uid_array = np.array(uid_lst)
        data_source_array = np.array(data_source_lst)

        for data_source in np.unique(data_source_array):
            pass_rates = defaultdict(list)

            data_source_mask = data_source_array == data_source
            is_correct_data_source = is_correct_array[data_source_mask]
            uids_data_source = uid_array[data_source_mask]

            for is_correct, uid in zip(
                is_correct_data_source, uids_data_source, strict=False
            ):
                pass_rates[uid].append(is_correct)

            metrics[f"val/{data_source}/pass@1"] = np.mean(is_correct_data_source)
            metrics[f"val/{data_source}/pass@{n_val_samples}"] = np.mean(
                [1 if any(pass_rate) else 0 for pass_rate in pass_rates.values()]
            )

            # Add workflow metrics for this data source
            if data_source in workflow_metrics_by_source:
                for key, values in workflow_metrics_by_source[data_source].items():
                    if values:  # Only add if we have values
                        metrics[f"val/{data_source}/{key}"] = np.mean(values)

        return metrics

    def generate_trajectories(self, batch, timing_raw=None, stop_condition=None, on_episode_done=None, min_rollouts=2, **kwargs):
        """
        Generates trajectories asynchronously using the agent execution engine's excute tasks method.
        Post-processing is done in the engine as well.

        Args:
            batch: The input batch for trajectory generation
            timing_raw: Dictionary to store timing information for profiling
            stop_condition: Optional callable for dynamic rollout early stopping.
            on_episode_done: Optional callable for dynamic rollout signal tracking.
            min_rollouts: Minimum rollouts per prompt (for progress display).
            **kwargs: Additional arguments to pass to trajectory_generator

        Returns:
            DataProto or (DataProto, list): Processed trajectories. When dynamic rollout
                is active, also returns the list of task_ids for selection.
        """
        if timing_raw is None:
            timing_raw = {}

        engine_kwargs = dict(kwargs)
        if stop_condition is not None:
            engine_kwargs["stop_condition"] = stop_condition
        if on_episode_done is not None:
            engine_kwargs["on_episode_done"] = on_episode_done
            engine_kwargs["min_rollouts"] = min_rollouts
        if "sampling_mode" in kwargs:
            engine_kwargs["sampling_mode"] = kwargs["sampling_mode"]

        with marked_timer("generate_trajectories", timing_raw, color="red"):
            coro = self.agent_execution_engine.execute_tasks_verl(batch, **engine_kwargs)
            final_gen_batch_output = asyncio.run_coroutine_threadsafe(
                coro, self._loop
            ).result()

        if stop_condition is not None:
            task_ids = batch.non_tensor_batch["task_ids"].tolist()
            return final_gen_batch_output, task_ids

        return final_gen_batch_output

    def _stepwise_advantage_broadcast(self, last_step_batch, non_last_step_batch):
        """
        Broadcast the advantage from last_step_batch to all other steps within the same episode and trajectory.
        """

        # NOTE: Currently takes the average of advantages. For GRPO, advantage and returns is uniform for each token so this makes no difference.
        # NOTE: For simplicity, assumes advantage and return is the same, which also holds for GRPO variants

        src_traj_ids = last_step_batch.non_tensor_batch["trajectory_ids"]
        src_eps_ids = last_step_batch.non_tensor_batch["episode_ids"]
        src_steps = last_step_batch.non_tensor_batch["step_nums"]
        src_mask = last_step_batch.batch["response_mask"]
        src_advantages = last_step_batch.batch["advantages"]

        tgt_traj_ids = non_last_step_batch.non_tensor_batch["trajectory_ids"]
        tgt_eps_ids = non_last_step_batch.non_tensor_batch["episode_ids"]
        tgt_mask = non_last_step_batch.batch["response_mask"]

        # Build id -> scalar advantage
        traj_ep_to_scalar_adv = {}
        for i, (traj_id, eps_id) in enumerate(
            zip(src_traj_ids, src_eps_ids, strict=False)
        ):
            mask = src_mask[i].bool()
            scalar = src_advantages[i][mask].mean()

            if self.config.rllm.stepwise_advantage.normalize_by_steps:
                # normalize the advantage against number of steps
                scalar = scalar / src_steps[i]
                # reassign the normalized advantage to last_step_batch as well
                last_step_batch.batch["advantages"][i][mask] = scalar

            traj_ep_to_scalar_adv[(traj_id, eps_id)] = scalar

        # Create new tensor for non_last_step_batch with per-token assignment
        scalar_rows = torch.stack(
            [
                torch.full_like(
                    tgt_mask[i],
                    fill_value=traj_ep_to_scalar_adv[(traj_id, eps_id)],
                    dtype=torch.float32,
                )
                for i, (traj_id, eps_id) in enumerate(
                    zip(tgt_traj_ids, tgt_eps_ids, strict=False)
                )
            ]
        )  # shape: (N2, T)

        # Apply the response mask of the target batch
        final_advantage = scalar_rows * tgt_mask

        # Assignment
        non_last_step_batch.batch["advantages"] = final_advantage
        non_last_step_batch.batch["returns"] = final_advantage

    def _pad_dataproto_to_world_size(self, batch):
        world_sizes = []
        if self.use_critic and self.critic_wg.world_size != 0:
            world_sizes.append(self.critic_wg.world_size)
        if self.use_reference_policy and self.ref_policy_wg.world_size != 0:
            world_sizes.append(self.ref_policy_wg.world_size)
        if self.use_rm and self.rm_wg.world_size != 0:
            world_sizes.append(self.rm_wg.world_size)
        if self.hybrid_engine:
            if self.actor_rollout_wg.world_size != 0:
                world_sizes.append(self.actor_rollout_wg.world_size)
        else:
            if self.actor_wg.world_size != 0:
                world_sizes.append(self.actor_wg.world_size)
            if hasattr(self, "rollout_wg") and self.rollout_wg.world_size != 0:
                world_sizes.append(self.rollout_wg.world_size)
        if not world_sizes:
            return batch

        world_size = reduce(math.lcm, world_sizes)

        batch = self._remove_padding(
            batch
        )  # Remove any padded steps from the batch (just in case)
        original_batch_size = batch.batch["prompts"].shape[0]
        batch, pad_size = pad_dataproto_to_divisor(batch, world_size)

        # for the padded dataproto, make the traj mask to 0. is_last_step also False
        for i in range(pad_size):
            idx = original_batch_size + i
            batch.non_tensor_batch["is_last_step"][idx] = False
            batch.non_tensor_batch["is_pad_step"][idx] = True
            batch.non_tensor_batch["is_valid"][idx] = False

        return batch

    def _remove_padding(self, batch):
        """Removes padded steps from the batch"""
        is_pad_step = batch.non_tensor_batch["is_pad_step"]
        non_pad_step_indices = np.where(is_pad_step == False)[0]
        batch = batch.select_idxs(
            non_pad_step_indices
        )  # This batch only has non_pad steps
        return batch

    def shutdown(self):
        """A cleanup method to gracefully stop the background event loop."""
        if (
            hasattr(self, "agent_execution_engine")
            and self.agent_execution_engine is not None
        ):
            self.agent_execution_engine.shutdown()
            self.agent_execution_engine = None
        if (
            hasattr(self, "_loop")
            and self._loop is not None
            and self._loop.is_running()
        ):
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, "_thread") and self._thread is not None:
            self._thread.join()

    def visualize_trajectory_last_step(self, tensor_batch, sample_idx=0, max_samples=1):
        """
        Visualize last steps from a workflow rollout using the shared visualization utility.
        """
        from rllm.utils.visualization import visualize_trajectories

        # Select only last steps if stepwise-advantage is enabled
        if "is_last_step" in tensor_batch.non_tensor_batch:
            is_last = tensor_batch.non_tensor_batch["is_last_step"]
            if is_last is not None and len(is_last) == len(tensor_batch):
                tensor_batch = tensor_batch[is_last]

        if len(tensor_batch) == 0:
            return

        end_idx = min(sample_idx + max_samples, len(tensor_batch))
        indices = list(range(sample_idx, end_idx))

        visualize_trajectories(
            batch=tensor_batch,
            tokenizer=self.tokenizer,
            sample_indices=indices,
            mask_key="response_mask",
            reward_key=(
                "step_rewards"
                if self.config.rllm.stepwise_advantage.enable
                and self.config.rllm.stepwise_advantage.mode == "per_step"
                else "traj_rewards"
            ),
            show_workflow_metadata=True,
        )


# ---------------------------------------------------------------------------
# Dynamic rollout helper functions
# ---------------------------------------------------------------------------

def _to_finite_float(value):
    try:
        reward = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(reward):
        return None
    return reward


def _get_reward_from_metrics(metrics):
    if not isinstance(metrics, dict):
        return None

    # Dynamic rollout should track the reward actually used for training. This
    # keeps partial-credit and future reward components visible to early stop
    # and diverse selection without hard-coding binary correctness semantics.
    for key in ("reward/outcome", "reward", "reward/accuracy"):
        reward = _to_finite_float(metrics.get(key))
        if reward is not None:
            return reward
    return None


def _get_episode_reward(episode):
    reward = _get_reward_from_metrics(getattr(episode, "metrics", None))
    if reward is not None:
        return reward
    if episode.trajectories:
        reward = _to_finite_float(episode.trajectories[0].reward)
        if reward is not None:
            return reward
    if episode.is_correct is not None:
        return 1.0 if episode.is_correct else 0.0
    return 0.0


def _get_dapo_signal_source(dapo_cfg):
    if dapo_cfg is None:
        return "reward"
    return normalize_dapo_signal_source(
        dapo_cfg.get("signal_source", dapo_cfg.get("signal_metric", "reward"))
    )


def _get_dynamic_expansion_policy(dynamic_cfg):
    if dynamic_cfg is None:
        return "until_signal"
    policy = str(dynamic_cfg.get("expansion_policy", "until_signal")).strip().lower()
    aliases = {
        "default": "until_signal",
        "current": "until_signal",
        "until_signal": "until_signal",
        "expand_until_signal": "until_signal",
        "signal": "until_signal",
        "expand_only_all_wrong": "expand_only_all_wrong",
        "all_wrong": "expand_only_all_wrong",
        "only_all_wrong": "expand_only_all_wrong",
    }
    if policy not in aliases:
        valid = ", ".join(sorted(set(aliases.values())))
        raise ValueError(
            f"dynamic_rollout.expansion_policy must be one of {valid}; got {policy!r}"
        )
    return aliases[policy]


def _get_correctness_from_metrics(metrics):
    if not isinstance(metrics, dict):
        return None
    reward = _to_finite_float(metrics.get("reward/accuracy"))
    if reward is None:
        return None
    return reward > 0.5


def _get_episode_correctness(episode):
    is_correct = getattr(episode, "is_correct", None)
    if is_correct is not None:
        return bool(is_correct)
    return _get_correctness_from_metrics(getattr(episode, "metrics", None))


def _get_episode_dapo_signal(episode, signal_source):
    if normalize_dapo_signal_source(signal_source) == "correctness":
        correctness = _get_episode_correctness(episode)
        return correctness if correctness is not None else False
    return _get_episode_reward(episode)


def _should_stop_dynamic_rollout(task_states, completed_rewards, min_rollouts, signal_threshold, completion_ratio_threshold=1.0):
    # All samples must have at least min_rollouts before any early stop is allowed
    for task_id, state in task_states.items():
        if state["completed"] < min_rollouts:
            return ""

    # Early stop if total completed rollouts reach the ratio threshold (e.g. 95%)
    total_completed = sum(s["completed"] for s in task_states.values())
    total_expected = sum(s["total_rollouts"] for s in task_states.values())
    if total_expected > 0 and total_completed / total_expected >= completion_ratio_threshold:
        return f"completion_ratio={total_completed/total_expected:.0%}>={completion_ratio_threshold:.0%}"

    num_with_signal = 0
    for task_id in task_states:
        rewards = completed_rewards.get(task_id, [])
        if len(rewards) >= 2 and np.std(rewards) > 0:
            num_with_signal += 1

    if len(task_states) == 0:
        return ""

    signal_ratio = num_with_signal / len(task_states)
    if signal_ratio >= signal_threshold:
        return f"signal_ratio={signal_ratio:.0%}>={signal_threshold:.0%}"
    return ""


def _select_diverse_rollouts(episodes_with_rewards, n_select):
    if len(episodes_with_rewards) <= n_select:
        return [idx for idx, _ in episodes_with_rewards]

    reward_groups = defaultdict(list)
    for idx, reward in episodes_with_rewards:
        reward_groups[reward].append(idx)

    selected = []
    for reward, indices in reward_groups.items():
        if len(selected) < n_select and indices:
            selected.append(indices.pop(0))

    remaining = []
    for indices in reward_groups.values():
        remaining.extend(indices)
    random.shuffle(remaining)
    selected.extend(remaining[:n_select - len(selected)])

    return selected[:n_select]


def _get_rounds_bucket(metrics):
    if not isinstance(metrics, dict):
        return "unknown"
    rounds = _to_finite_float(metrics.get("rounds"))
    if rounds is None:
        rounds = _to_finite_float(metrics.get("num_rounds"))
    if rounds is None:
        return "unknown"
    if rounds <= 3:
        return "short"
    if rounds <= 8:
        return "mid"
    return "long"


def _get_metric_int(metrics, *keys):
    if not isinstance(metrics, dict):
        return None
    for key in keys:
        value = _to_finite_float(metrics.get(key))
        if value is not None:
            return int(round(value))
    return None


def _get_expanded_tool_calls(metrics):
    total = _get_metric_int(metrics, "tool_calls_expanded", "total_tool_calls_expanded")
    if total is not None:
        return total

    if isinstance(metrics, dict):
        expanded_sum = 0
        has_expanded = False
        for name in ("text_search", "image_search", "visit", "zoom_in", "crop_and_search"):
            count = _get_metric_int(metrics, f"{name}_calls_expanded")
            if count is not None:
                expanded_sum += count
                has_expanded = True
        if has_expanded:
            return expanded_sum

    return _get_metric_int(metrics, "tool_calls", "num_tool_calls", "total_tool_calls")


def _get_behavior_signature(metrics):
    if not isinstance(metrics, dict):
        return "unknown"
    rounds_bucket = _get_rounds_bucket(metrics)
    signature = metrics.get("tool_sequence_key")
    if signature:
        return f"tools={signature}|rounds={rounds_bucket}"

    tool_parts = []
    for name in ("text_search", "image_search", "visit", "zoom_in", "crop_and_search"):
        count = metrics.get(f"{name}_calls")
        if count is not None:
            try:
                count = int(count)
            except (TypeError, ValueError):
                count = 0
            if count > 0:
                tool_parts.append(f"{name}:{count}")
    if tool_parts:
        return f"tools={'|'.join(tool_parts)}|rounds={rounds_bucket}"

    completed = metrics.get("completed")
    return f"tools=unknown|rounds={rounds_bucket}|completed={completed}"


def _get_round_tool_behavior_signature(metrics):
    if not isinstance(metrics, dict):
        return "unknown"
    rounds = _get_metric_int(metrics, "rounds", "num_rounds")
    rounds_part = str(rounds) if rounds is not None else _get_rounds_bucket(metrics)
    tool_calls = _get_expanded_tool_calls(metrics)
    tool_part = str(tool_calls) if tool_calls is not None else "unknown"
    return f"rounds={rounds_part}|tool_calls_expanded={tool_part}"


def _select_reward_behavior_diverse_rollouts(episodes_with_rewards_and_behavior, n_select):
    if len(episodes_with_rewards_and_behavior) <= n_select:
        return [idx for idx, _, _ in episodes_with_rewards_and_behavior]

    reward_groups = defaultdict(list)
    for idx, reward, behavior in episodes_with_rewards_and_behavior:
        reward_groups[reward].append((idx, behavior))

    selected = []
    selected_set = set()
    seen_behaviors = set()

    def take_from_group(items, prefer_new_behavior=True):
        for pos, (idx, behavior) in enumerate(items):
            if idx in selected_set:
                continue
            if prefer_new_behavior and behavior in seen_behaviors:
                continue
            items.pop(pos)
            return idx, behavior
        if prefer_new_behavior:
            return None
        for pos, (idx, behavior) in enumerate(items):
            if idx not in selected_set:
                items.pop(pos)
                return idx, behavior
        return None

    while len(selected) < n_select:
        progressed = False
        for reward, items in reward_groups.items():
            if len(selected) >= n_select:
                break
            picked = take_from_group(items, prefer_new_behavior=True)
            if picked is None:
                continue
            idx, behavior = picked
            selected.append(idx)
            selected_set.add(idx)
            seen_behaviors.add(behavior)
            progressed = True
        if progressed:
            continue
        for reward, items in reward_groups.items():
            if len(selected) >= n_select:
                break
            picked = take_from_group(items, prefer_new_behavior=False)
            if picked is None:
                continue
            idx, behavior = picked
            selected.append(idx)
            selected_set.add(idx)
            seen_behaviors.add(behavior)
            progressed = True
        if not progressed:
            break

    remaining = []
    for items in reward_groups.values():
        remaining.extend(idx for idx, _ in items if idx not in selected_set)
    random.shuffle(remaining)
    selected.extend(remaining[:n_select - len(selected)])

    return selected[:n_select]


def _dapo_group_has_signal(values, reward_std_threshold, signal_source="reward"):
    return dapo_group_has_signal(values, reward_std_threshold, signal_source)


def _group_key_from_episode_id(episode_id):
    episode_id = str(episode_id)
    return episode_id.rsplit(":", 1)[0] if ":" in episode_id else episode_id


def _compute_group_reward_stats(batch):
    episode_ids = batch.non_tensor_batch["episode_ids"]
    task_ids = batch.non_tensor_batch.get("task_ids", None)
    rewards_by_group = defaultdict(list)
    seen_episode_ids = set()

    for row_idx, episode_id in enumerate(episode_ids):
        if episode_id in seen_episode_ids:
            continue
        seen_episode_ids.add(episode_id)
        if task_ids is not None:
            group_key = str(task_ids[row_idx])
        else:
            group_key = _group_key_from_episode_id(episode_id)
        rewards_by_group[group_key].append(_episode_reward_from_batch_row(batch, row_idx))

    stats = {}
    for group_key, rewards in rewards_by_group.items():
        if not rewards:
            continue
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        if rewards_tensor.numel() == 1:
            mean = torch.tensor(0.0)
            std = torch.tensor(1.0)
        else:
            mean = rewards_tensor.mean()
            std = rewards_tensor.std()
        stats[str(group_key)] = (float(mean.item()), float(std.item()), int(rewards_tensor.numel()))
    return stats


def _attach_full_rollout_advantage_stats(batch, stats):
    task_ids = batch.non_tensor_batch.get("task_ids", None)
    episode_ids = batch.non_tensor_batch.get("episode_ids", None)
    means = []
    stds = []
    counts = []
    for row_idx in range(len(batch.batch["input_ids"])):
        if task_ids is not None:
            group_key = str(task_ids[row_idx])
        else:
            group_key = _group_key_from_episode_id(episode_ids[row_idx])
        mean, std, count = stats.get(group_key, (0.0, 1.0, 0))
        means.append(mean)
        stds.append(std)
        counts.append(count)
    batch.non_tensor_batch["full_rollout_advantage_mean"] = np.array(means, dtype=np.float32)
    batch.non_tensor_batch["full_rollout_advantage_std"] = np.array(stds, dtype=np.float32)
    batch.non_tensor_batch["full_rollout_advantage_count"] = np.array(counts, dtype=np.int32)
    return batch


def _apply_full_rollout_rloo_advantage(batch):
    scores = batch.batch["token_level_rewards"].sum(dim=-1)
    means = torch.as_tensor(
        batch.non_tensor_batch["full_rollout_advantage_mean"],
        dtype=scores.dtype,
        device=scores.device,
    )
    counts = torch.as_tensor(
        batch.non_tensor_batch["full_rollout_advantage_count"],
        dtype=scores.dtype,
        device=scores.device,
    )
    scalars = torch.zeros_like(scores)
    valid = counts > 1
    scalars[valid] = (
        scores[valid] * counts[valid] / (counts[valid] - 1)
        - means[valid] * counts[valid] / (counts[valid] - 1)
    )
    advantages = scalars.unsqueeze(-1) * batch.batch["response_mask"]
    batch.batch["advantages"] = advantages
    batch.batch["returns"] = advantages
    return batch


def _select_dapo_final_uids(unique_uids, target_groups, selection_strategy):
    unique_uids = list(unique_uids)
    if len(unique_uids) <= target_groups:
        return unique_uids
    if selection_strategy == "random":
        return np.random.choice(unique_uids, size=target_groups, replace=False)
    return unique_uids[:target_groups]


def _episode_reward_from_batch_row(batch, row_idx):
    metrics = batch.non_tensor_batch.get("metrics", None)
    if metrics is not None:
        reward = _get_reward_from_metrics(metrics[row_idx])
        if reward is not None:
            return reward
    reward_tensor = batch.batch.get("traj_rewards", None)
    if reward_tensor is not None:
        try:
            return float(reward_tensor[row_idx].sum().detach().cpu().item())
        except Exception:
            pass
    is_correct = batch.non_tensor_batch.get("is_correct", None)
    if is_correct is not None:
        return 1.0 if bool(is_correct[row_idx]) else 0.0
    return 0.0


def _episode_correctness_from_batch_row(batch, row_idx):
    is_correct = batch.non_tensor_batch.get("is_correct", None)
    if is_correct is not None:
        return bool(is_correct[row_idx])
    metrics = batch.non_tensor_batch.get("metrics", None)
    if metrics is not None:
        return _get_correctness_from_metrics(metrics[row_idx])
    return None


def _episode_dapo_signal_from_batch_row(batch, row_idx, signal_source):
    if normalize_dapo_signal_source(signal_source) == "correctness":
        correctness = _episode_correctness_from_batch_row(batch, row_idx)
        return correctness if correctness is not None else False
    return _episode_reward_from_batch_row(batch, row_idx)


def _merge_dapo_generated_groups(gen_output, original_batch):
    episode_ids = gen_output.non_tensor_batch["episode_ids"]
    new_repeat_counts = []
    prev_eid = None
    for eid in episode_ids:
        if eid != prev_eid:
            new_repeat_counts.append(1)
            prev_eid = eid
        else:
            new_repeat_counts[-1] += 1
    gen_output.meta_info["repeat_counts"] = new_repeat_counts

    selected_episode_order = []
    seen_selected = set()
    for eid in episode_ids:
        if eid not in seen_selected:
            selected_episode_order.append(eid)
            seen_selected.add(eid)

    task_id_arr = original_batch.non_tensor_batch["task_ids"]
    group_keys = [eid.rsplit(":", 1)[0] if ":" in eid else eid for eid in selected_episode_order]
    group_indices_by_task_id = {
        group_key: np.where(task_id_arr == group_key)[0].tolist()
        for group_key in set(group_keys)
    }
    next_group_offset = defaultdict(int)
    selected_indices = []
    for eid in selected_episode_order:
        group_key = eid.rsplit(":", 1)[0] if ":" in eid else eid
        group_indices = group_indices_by_task_id.get(group_key, [])
        rollout_idx = None
        if ":" in eid:
            try:
                rollout_idx = int(eid.rsplit(":", 1)[1])
            except ValueError:
                rollout_idx = None

        if rollout_idx is not None and rollout_idx < len(group_indices):
            original_idx = group_indices[rollout_idx]
        else:
            offset = next_group_offset[group_key]
            if offset >= len(group_indices):
                raise ValueError(
                    f"No original batch row available for DAPO episode {eid}"
                )
            original_idx = group_indices[offset]
        next_group_offset[group_key] += 1
        selected_indices.append(original_idx)

    original_batch = original_batch.select_idxs(selected_indices)
    original_batch = original_batch.sample_level_repeat(new_repeat_counts)
    gen_output.meta_info.pop("repeat_counts", None)
    return original_batch.union(gen_output)


def _select_batch_rollouts_per_group(batch, min_rollouts, selection_strategy):
    """Select up to min_rollouts episodes per selected DAPO group from a unioned DataProto."""
    episode_ids = batch.non_tensor_batch["episode_ids"]
    task_ids = batch.non_tensor_batch["task_ids"]
    metrics = batch.non_tensor_batch.get("metrics", None)

    episode_info = {}
    sample_episodes = defaultdict(list)
    for row_idx in range(len(episode_ids)):
        eid = episode_ids[row_idx]
        if eid in episode_info:
            continue
        row_metrics = metrics[row_idx] if metrics is not None else None
        group_key = str(task_ids[row_idx])
        reward = _episode_reward_from_batch_row(batch, row_idx)
        behavior_signature = _get_round_tool_behavior_signature(row_metrics)
        episode_info[eid] = (row_idx, reward, group_key, behavior_signature)
        sample_episodes[group_key].append((eid, reward, behavior_signature))

    selected_episode_ids = set()
    sample_selected_count = {}
    sample_selected_ids = {}
    unique_groups = list(sample_episodes.keys())
    for group_key in unique_groups:
        episodes = sample_episodes[group_key]
        if len(episodes) <= min_rollouts:
            selected = [eid for eid, _, _ in episodes]
        elif selection_strategy == "diverse":
            selected = _select_diverse_rollouts(
                [(eid, reward) for eid, reward, _ in episodes], min_rollouts,
            )
        elif selection_strategy == "reward_behavior_diverse":
            selected = _select_reward_behavior_diverse_rollouts(
                [(eid, reward, behavior) for eid, reward, behavior in episodes],
                min_rollouts,
            )
        elif selection_strategy == "random":
            indices = list(range(len(episodes)))
            random.shuffle(indices)
            selected = [episodes[i][0] for i in indices[:min_rollouts]]
        else:
            selected = [eid for eid, _, _ in episodes[:min_rollouts]]
        selected_episode_ids.update(selected)
        sample_selected_count[group_key] = len(selected)
        sample_selected_ids[group_key] = list(selected)

    selected_mask = np.isin(episode_ids, list(selected_episode_ids))
    batch = batch[selected_mask]
    n_rollouts = min(sample_selected_count.values()) if sample_selected_count else min_rollouts

    selection_info = {
        "selection_strategy": selection_strategy,
        "min_rollouts": int(min_rollouts),
        "n_rollouts": int(n_rollouts),
        "selected_episode_ids": sorted(selected_episode_ids),
        "discarded_episode_ids": sorted(set(episode_info.keys()) - selected_episode_ids),
        "per_sample": {
            group_key: {
                "selected": sample_selected_count.get(group_key, 0),
                "discarded": len(sample_episodes[group_key]) - sample_selected_count.get(group_key, 0),
                "total": len(sample_episodes[group_key]),
                "selected_episode_ids": sorted(sample_selected_ids.get(group_key, [])),
                "discarded_episode_ids": sorted(
                    eid
                    for eid, _, _ in sample_episodes[group_key]
                    if eid not in set(sample_selected_ids.get(group_key, []))
                ),
            }
            for group_key in unique_groups
        },
    }
    return batch, n_rollouts, selection_info


def _filter_dapo_groups_after_reward(
    batch, target_groups, reward_std_threshold, final_selection_strategy,
    signal_source="reward", required_rollouts=None,
):
    signal_source = normalize_dapo_signal_source(signal_source)
    required_rollouts = int(required_rollouts) if required_rollouts is not None else None
    episode_ids = batch.non_tensor_batch["episode_ids"]
    task_ids = batch.non_tensor_batch["task_ids"]

    episode_info = {}
    group_episode_ids = defaultdict(list)
    for row_idx in range(len(episode_ids)):
        eid = episode_ids[row_idx]
        if eid in episode_info:
            continue
        group_key = str(task_ids[row_idx])
        signal = _episode_dapo_signal_from_batch_row(batch, row_idx, signal_source)
        episode_info[eid] = (row_idx, signal, group_key)
        group_episode_ids[group_key].append(eid)

    valid_group_ids = []
    invalid_group_ids = []
    for group_key, eids in group_episode_ids.items():
        signals = [episode_info[eid][1] for eid in eids]
        if required_rollouts is not None and len(eids) < required_rollouts:
            invalid_group_ids.append(group_key)
        elif _dapo_group_has_signal(signals, reward_std_threshold, signal_source):
            valid_group_ids.append(group_key)
        else:
            invalid_group_ids.append(group_key)

    selected_group_ids = _select_dapo_final_uids(
        valid_group_ids, target_groups, final_selection_strategy,
    )
    selected_group_ids = [str(group_id) for group_id in selected_group_ids]
    selected_episode_ids = {
        eid
        for group_id in selected_group_ids
        for eid in group_episode_ids.get(group_id, [])
    }
    selected_mask = np.isin(task_ids, selected_group_ids)
    batch = batch[selected_mask]

    selection_info = {
        "selection_strategy": f"dapo_group_filter_after_{signal_source}_signal",
        "target_groups": int(target_groups),
        "signal_source": signal_source,
        "reward_std_threshold": float(reward_std_threshold),
        "required_rollouts": None if required_rollouts is None else int(required_rollouts),
        "selected_group_ids": list(selected_group_ids),
        "valid_group_ids": list(valid_group_ids),
        "invalid_group_ids": list(invalid_group_ids),
        "selected_episode_ids": sorted(selected_episode_ids),
        "discarded_episode_ids": sorted(set(episode_info.keys()) - selected_episode_ids),
        "per_sample": {
            group_key: {
                "selected": sum(1 for eid in group_episode_ids[group_key] if eid in selected_episode_ids),
                "discarded": sum(1 for eid in group_episode_ids[group_key] if eid not in selected_episode_ids),
                "selected_group": int(group_key in selected_group_ids),
                "valid_signal": int(group_key in valid_group_ids),
                "total": len(group_episode_ids[group_key]),
                "selected_episode_ids": sorted(eid for eid in group_episode_ids[group_key] if eid in selected_episode_ids),
                "discarded_episode_ids": sorted(eid for eid in group_episode_ids[group_key] if eid not in selected_episode_ids),
            }
            for group_key in group_episode_ids.keys()
        },
    }
    return batch, selection_info


def _select_dapo_groups(
    gen_output, original_batch, selected_group_ids, target_groups,
    reward_std_threshold, signal_source="reward", required_rollouts=None,
):
    """Keep complete DAPO groups that have signal under the configured source."""
    signal_source = normalize_dapo_signal_source(signal_source)
    episode_ids = gen_output.non_tensor_batch["episode_ids"]
    is_correct = gen_output.non_tensor_batch.get("is_correct", None)
    metrics = gen_output.non_tensor_batch.get("metrics", None)
    task_id_arr = original_batch.non_tensor_batch["task_ids"]
    expected_group_counts = Counter(str(task_id) for task_id in task_id_arr)
    required_rollouts = int(required_rollouts) if required_rollouts is not None else None

    episode_info = {}
    group_episode_ids = defaultdict(list)
    for row_idx in range(len(episode_ids)):
        eid = episode_ids[row_idx]
        if eid in episode_info:
            continue
        row_metrics = metrics[row_idx] if metrics is not None else None
        if signal_source == "correctness":
            signal = bool(is_correct[row_idx]) if is_correct is not None else _get_correctness_from_metrics(row_metrics)
            if signal is None:
                signal = False
        else:
            signal = _get_reward_from_metrics(row_metrics)
            if signal is None:
                signal = 1.0 if is_correct is not None and is_correct[row_idx] else 0.0
        group_key = eid.rsplit(":", 1)[0] if ":" in eid else eid
        episode_info[eid] = (row_idx, signal, group_key)
        group_episode_ids[group_key].append(eid)

    valid_group_ids = []
    invalid_group_ids = []
    for group_key, eids in group_episode_ids.items():
        signals = [episode_info[eid][1] for eid in eids]
        group_required_rollouts = required_rollouts
        if group_required_rollouts is None:
            group_required_rollouts = expected_group_counts.get(str(group_key), 0)
        if group_required_rollouts and len(eids) < group_required_rollouts:
            invalid_group_ids.append(group_key)
        elif _dapo_group_has_signal(signals, reward_std_threshold, signal_source):
            valid_group_ids.append(group_key)
        else:
            invalid_group_ids.append(group_key)

    valid_group_id_set = set(valid_group_ids)
    selected_group_ids = [
        group_id for group_id in selected_group_ids
        if group_id in valid_group_id_set
    ]
    if not selected_group_ids:
        selected_group_ids = valid_group_ids[:target_groups]
    else:
        selected_group_ids = selected_group_ids[:target_groups]

    selected_episode_ids = {
        eid
        for group_id in selected_group_ids
        for eid in group_episode_ids.get(group_id, [])
    }
    kept_row_indices = [
        i for i in range(len(episode_ids))
        if episode_ids[i] in selected_episode_ids
    ]
    gen_output = gen_output.select_idxs(kept_row_indices)

    new_episode_ids = gen_output.non_tensor_batch["episode_ids"]
    new_repeat_counts = []
    prev_eid = None
    for eid in new_episode_ids:
        if eid != prev_eid:
            new_repeat_counts.append(1)
            prev_eid = eid
        else:
            new_repeat_counts[-1] += 1
    gen_output.meta_info["repeat_counts"] = new_repeat_counts

    selected_episode_order = []
    seen_selected = set()
    for eid in new_episode_ids:
        if eid not in seen_selected:
            selected_episode_order.append(eid)
            seen_selected.add(eid)

    group_indices_by_task_id = {
        group_key: np.where(task_id_arr == group_key)[0].tolist()
        for group_key in group_episode_ids.keys()
    }
    next_group_offset = defaultdict(int)
    truncated_indices = []
    for eid in selected_episode_order:
        group_key = eid.rsplit(":", 1)[0] if ":" in eid else eid
        group_indices = group_indices_by_task_id.get(group_key, [])
        rollout_idx = None
        if ":" in eid:
            try:
                rollout_idx = int(eid.rsplit(":", 1)[1])
            except ValueError:
                rollout_idx = None

        if rollout_idx is not None and rollout_idx < len(group_indices):
            original_idx = group_indices[rollout_idx]
        else:
            offset = next_group_offset[group_key]
            if offset >= len(group_indices):
                raise ValueError(
                    f"No original batch row available for selected DAPO episode {eid}"
                )
            original_idx = group_indices[offset]
        next_group_offset[group_key] += 1
        truncated_indices.append(original_idx)

    original_batch = original_batch.select_idxs(truncated_indices)
    original_batch = original_batch.sample_level_repeat(new_repeat_counts)
    gen_output.meta_info.pop("repeat_counts", None)
    original_batch = original_batch.union(gen_output)

    selection_info = {
        "selection_strategy": f"dapo_group_filter_after_{signal_source}_signal",
        "target_groups": int(target_groups),
        "signal_source": signal_source,
        "reward_std_threshold": float(reward_std_threshold),
        "required_rollouts": None if required_rollouts is None else int(required_rollouts),
        "selected_group_ids": list(selected_group_ids),
        "valid_group_ids": list(valid_group_ids),
        "invalid_group_ids": list(invalid_group_ids),
        "selected_episode_ids": sorted(selected_episode_ids),
        "discarded_episode_ids": sorted(set(episode_info.keys()) - selected_episode_ids),
        "per_sample": {
            group_key: {
                "selected": sum(1 for eid in group_episode_ids[group_key] if eid in selected_episode_ids),
                "discarded": sum(1 for eid in group_episode_ids[group_key] if eid not in selected_episode_ids),
                "selected_group": int(group_key in selected_group_ids),
                "valid_signal": int(group_key in valid_group_ids),
                "total": len(group_episode_ids[group_key]),
                "selected_episode_ids": sorted(eid for eid in group_episode_ids[group_key] if eid in selected_episode_ids),
                "discarded_episode_ids": sorted(eid for eid in group_episode_ids[group_key] if eid not in selected_episode_ids),
            }
            for group_key in group_episode_ids.keys()
        },
    }

    return gen_output, original_batch, selection_info


def _select_and_truncate(gen_output, original_batch, min_rollouts, selection_strategy):
    """Select min_rollouts episodes per sample, truncate gen_output and original_batch.

    Groups episodes by parsing the task_id prefix from episode_ids (format: "task_id:rollout_idx").
    """
    episode_ids = gen_output.non_tensor_batch["episode_ids"]
    is_correct = gen_output.non_tensor_batch["is_correct"]
    metrics = gen_output.non_tensor_batch.get("metrics", None)

    # Build mapping: episode_id -> (first_row_idx, reward, group_key, behavior_signature)
    # episode_id format is "task_id:rollout_idx", group by task_id
    episode_info = {}
    for row_idx in range(len(episode_ids)):
        eid = episode_ids[row_idx]
        if eid not in episode_info:
            row_metrics = metrics[row_idx] if metrics is not None else None
            reward = _get_reward_from_metrics(row_metrics)
            if reward is None:
                reward = 1.0 if is_correct[row_idx] else 0.0
            group_key = eid.rsplit(":", 1)[0] if ":" in eid else eid
            behavior_signature = _get_behavior_signature(row_metrics)
            episode_info[eid] = (row_idx, reward, group_key, behavior_signature)

    # Group episodes by sample (task_id prefix)
    sample_episodes = defaultdict(list)
    for eid, (row_idx, reward, group_key, behavior_signature) in episode_info.items():
        sample_episodes[group_key].append((eid, reward, behavior_signature))

    # Select min_rollouts episodes per sample
    selected_episode_ids = set()
    sample_selected_count = {}
    sample_selected_ids = {}

    unique_groups = list(sample_episodes.keys())
    for group_key in unique_groups:
        episodes = sample_episodes[group_key]
        if len(episodes) <= min_rollouts:
            selected = [eid for eid, _, _ in episodes]
        elif selection_strategy == "diverse":
            selected = _select_diverse_rollouts(
                [(eid, r) for eid, r, _ in episodes], min_rollouts
            )
        elif selection_strategy == "reward_behavior_diverse":
            selected = _select_reward_behavior_diverse_rollouts(
                [(eid, r, b) for eid, r, b in episodes], min_rollouts
            )
        elif selection_strategy == "random":
            indices = list(range(len(episodes)))
            random.shuffle(indices)
            selected = [episodes[i][0] for i in indices[:min_rollouts]]
        else:  # "first"
            selected = [eid for eid, _, _ in episodes[:min_rollouts]]
        selected_episode_ids.update(selected)
        sample_selected_count[group_key] = len(selected)
        sample_selected_ids[group_key] = list(selected)

    n_rollouts = min(sample_selected_count.values()) if sample_selected_count else min_rollouts

    # Keep only step rows belonging to selected episodes
    kept_row_indices = [
        i for i in range(len(episode_ids))
        if episode_ids[i] in selected_episode_ids
    ]
    gen_output = gen_output.select_idxs(kept_row_indices)

    # Rebuild repeat_counts
    new_episode_ids = gen_output.non_tensor_batch["episode_ids"]
    new_repeat_counts = []
    prev_eid = None
    for eid in new_episode_ids:
        if eid != prev_eid:
            new_repeat_counts.append(1)
            prev_eid = eid
        else:
            new_repeat_counts[-1] += 1
    gen_output.meta_info["repeat_counts"] = new_repeat_counts

    # Truncate original_batch to match selected episodes in the exact gen_output
    # episode order. Dynamic rollout may schedule tasks round-major (A, B, A, B)
    # while completed episodes are returned sample-major (A..., B...). Sorting the
    # original indices would silently misalign sample ids with is_correct.
    selected_episode_order = []
    seen_selected = set()
    for eid in new_episode_ids:
        if eid not in seen_selected:
            selected_episode_order.append(eid)
            seen_selected.add(eid)

    task_id_arr = original_batch.non_tensor_batch["task_ids"]
    group_indices_by_task_id = {
        group_key: np.where(task_id_arr == group_key)[0].tolist()
        for group_key in unique_groups
    }
    next_group_offset = defaultdict(int)
    truncated_indices = []
    for eid in selected_episode_order:
        group_key = eid.rsplit(":", 1)[0] if ":" in eid else eid
        group_indices = group_indices_by_task_id.get(group_key, [])
        rollout_idx = None
        if ":" in eid:
            try:
                rollout_idx = int(eid.rsplit(":", 1)[1])
            except ValueError:
                rollout_idx = None

        if rollout_idx is not None and rollout_idx < len(group_indices):
            original_idx = group_indices[rollout_idx]
        else:
            offset = next_group_offset[group_key]
            if offset >= len(group_indices):
                raise ValueError(
                    f"No original batch row available for selected episode {eid}"
                )
            original_idx = group_indices[offset]
        next_group_offset[group_key] += 1
        truncated_indices.append(original_idx)

    original_batch = original_batch.select_idxs(truncated_indices)

    # Apply sample_level_repeat and union
    original_batch = original_batch.sample_level_repeat(new_repeat_counts)
    gen_output.meta_info.pop("repeat_counts", None)
    original_batch = original_batch.union(gen_output)

    # Build selection manifest for visualization
    selection_info = {
        "selection_strategy": selection_strategy,
        "selected_episode_ids": sorted(selected_episode_ids),
        "discarded_episode_ids": sorted(set(episode_info.keys()) - selected_episode_ids),
        "per_sample": {
            gk: {
                "selected": sample_selected_count.get(gk, 0),
                "discarded": len(sample_episodes[gk]) - sample_selected_count.get(gk, 0),
                "total": len(sample_episodes[gk]),
                "selected_episode_ids": sorted(sample_selected_ids.get(gk, [])),
                "discarded_episode_ids": sorted(
                    eid
                    for eid, _, _ in sample_episodes[gk]
                    if eid not in set(sample_selected_ids.get(gk, []))
                ),
            }
            for gk in unique_groups
        },
    }

    return gen_output, original_batch, n_rollouts, selection_info
