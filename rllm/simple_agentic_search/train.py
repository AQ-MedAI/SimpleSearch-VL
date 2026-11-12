import ast
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import hydra
import yaml
from omegaconf import OmegaConf

from simple_agentic_search.config_validation import resolve_agent_api
from simple_agentic_search.tools import get_all_tools
from simple_agentic_search.deepresearch_workflow import DeepResearchWorkflow

from rllm.data.dataset import DatasetRegistry
from rllm.rewards.reward_fn import deepresearch_reward_fn_async

# Configure logging: suppress verbose output, keep only warnings+ on terminal
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("simple_agentic_search.deepresearch_workflow").setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def _patch_image_processor_pixels():
    """Patch Qwen VL image processor to respect MAX_PIXELS/MIN_PIXELS env vars.

    VDR's verl doesn't read these env vars natively (unlike MARS's verl),
    so we monkey-patch the processor class at import time.
    """
    max_pixels = int(os.getenv("MAX_PIXELS", "0"))
    min_pixels = int(os.getenv("MIN_PIXELS", "0"))
    if not max_pixels and not min_pixels:
        return

    try:
        from transformers import AutoImageProcessor
        _orig_from_pretrained = AutoImageProcessor.from_pretrained

        @classmethod
        def _patched_from_pretrained(cls, *args, **kwargs):
            processor = _orig_from_pretrained.__func__(cls, *args, **kwargs)
            if max_pixels and hasattr(processor, "max_pixels"):
                processor.max_pixels = max_pixels
                logger.warning("Patched image processor max_pixels=%d", max_pixels)
            if min_pixels and hasattr(processor, "min_pixels"):
                processor.min_pixels = min_pixels
                logger.warning("Patched image processor min_pixels=%d", min_pixels)
            return processor

        AutoImageProcessor.from_pretrained = _patched_from_pretrained

        # Also patch AutoProcessor which wraps AutoImageProcessor
        from transformers import AutoProcessor
        _orig_auto_from_pretrained = AutoProcessor.from_pretrained

        @classmethod
        def _patched_auto_from_pretrained(cls, *args, **kwargs):
            processor = _orig_auto_from_pretrained.__func__(cls, *args, **kwargs)
            if hasattr(processor, "image_processor"):
                ip = processor.image_processor
                if max_pixels and hasattr(ip, "max_pixels"):
                    ip.max_pixels = max_pixels
                if min_pixels and hasattr(ip, "min_pixels"):
                    ip.min_pixels = min_pixels
                logger.warning("Patched AutoProcessor.image_processor pixels: max=%s min=%s", max_pixels, min_pixels)
            return processor

        AutoProcessor.from_pretrained = _patched_auto_from_pretrained
    except ImportError:
        pass


_patch_image_processor_pixels()


def _patch_verl_parquet(name: str, split: str):
    """Promote extra_info fields to top-level parquet columns in one read-write pass."""
    import pandas as pd
    dataset_dir = os.path.join(DatasetRegistry._DATASET_DIR, name)
    verl_path = os.path.join(dataset_dir, f"{split}_verl.parquet")
    df = pd.read_parquet(verl_path)
    df["data_source"] = df["extra_info"].apply(
        lambda x: x.get("data_source") if isinstance(x, dict) else None
    )
    df["sample_idx"] = df["extra_info"].apply(
        lambda x: x.get("sample_idx", -1) if isinstance(x, dict) else -1
    )
    df["id"] = df["extra_info"].apply(
        lambda x: str(x.get("id", "")) if isinstance(x, dict) else ""
    )
    df.to_parquet(verl_path)
    logger.warning("Patched %s with [data_source, sample_idx, id] columns (%d rows)", verl_path, len(df))


def _load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file and return list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _format_samples(data: list[dict], start_id: int = 0) -> list[dict]:
    """Convert raw JSONL records to the format expected by DatasetRegistry."""
    output = []
    for i, item in enumerate(data):
        source_path = item.get("data_source_path") or item.get("source_path") or item.get("source")
        source_name = item.get("data_source") or item.get("dataset_name")
        if source_name is None and source_path:
            source_name = Path(source_path).stem

        sample_id = item.get("id")
        if sample_id is None or str(sample_id) == "":
            sample_id = start_id + i

        output.append({
            "id": str(sample_id),
            "question": f"image_id: 1 \n Question: {item['question']}",
            "answer": item["answer"],
            "images": item.get("images", []),
            "data_source": source_name,
            "data_source_path": source_path,
            "sample_idx": item.get("sample_idx", -1),
        })
    return output


def _to_plain_container(value):
    if value is None:
        return None
    try:
        return OmegaConf.to_container(value, resolve=True)
    except Exception:
        return value


def _normalize_path_list(value, field_name: str) -> list[str]:
    """Normalize Hydra/YAML path fields into a list of path strings.

    run.sh forwards list values through Hydra overrides. Depending on quoting,
    OmegaConf may surface them as a ListConfig, a plain list, or a string such
    as "['a.jsonl']". Keep the loader tolerant so a valid YAML list is never
    treated as one literal filename.
    """
    value = _to_plain_container(value)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            for parser in (ast.literal_eval, yaml.safe_load):
                try:
                    parsed = parser(text)
                except Exception:
                    continue
                if isinstance(parsed, (list, tuple)):
                    return [str(item) for item in parsed if str(item)]
                if isinstance(parsed, str) and parsed != text:
                    return _normalize_path_list(parsed, field_name)
        return [text]
    logger.warning("Coercing %s=%r to a single path string", field_name, value)
    return [str(value)]


def _load_eval_tasks_from_files(test_files: list[str]) -> list[dict]:
    """Load eval JSONL files directly for API-model evaluation."""
    all_test = []
    for path in test_files:
        samples = _load_jsonl(path)
        source_name = Path(path).stem
        for idx, sample in enumerate(samples):
            sample.setdefault("data_source", source_name)
            sample.setdefault("data_source_path", path)
            sample["sample_idx"] = idx
        logger.warning("Loaded %d eval samples from %s", len(samples), path)
        all_test.extend(samples)
    return _format_samples(all_test)


def _build_common_workflow_args(config, wf_args, rollout_save_dir: str, rollout_batch_size: int) -> dict:
    tools_config = wf_args.get("tools_config", None)
    if tools_config is not None:
        tools_config = dict(_to_plain_container(tools_config))

    enabled_tools = None
    if tools_config is not None:
        enabled_tools = tools_config.get("enabled_tools", None)
        if enabled_tools is not None:
            enabled_tools = list(enabled_tools)

    reward_config = wf_args.get("reward", None)
    if reward_config is None:
        reward_config = config.get("reward", None)
    if reward_config is not None:
        reward_config = _to_plain_container(reward_config)

    agent_config = wf_args.get("agent_config", None)
    if agent_config is not None:
        agent_config = dict(_to_plain_container(agent_config))

    return {
        "reward_function": deepresearch_reward_fn_async,
        "tools": get_all_tools(tools_config),
        "rollout_save_dir": rollout_save_dir,
        "rollout_batch_size": rollout_batch_size,
        "enabled_tools": enabled_tools,
        "reward_config": reward_config,
        "tools_config": tools_config,
        "agent_config": agent_config,
        "timeout": wf_args.get("timeout", 1000000.0),
    }


def _filter_eval_metrics_for_output(val_metrics: dict) -> dict:
    return {
        k: v
        for k, v in val_metrics.items()
        if k.endswith("/pass@1") or k.endswith("/completed")
    }


def _save_eval_metrics(default_local_dir: str, val_metrics: dict, step: int) -> None:
    eval_dir = os.path.join(default_local_dir, "eval_results")
    os.makedirs(eval_dir, exist_ok=True)
    serializable_metrics = {
        k: float(v.item()) if hasattr(v, "item") else float(v) if isinstance(v, (int, float)) else v
        for k, v in _filter_eval_metrics_for_output(val_metrics).items()
    }
    serializable_metrics["step"] = step
    eval_file = os.path.join(eval_dir, f"eval_step_{step}.json")
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
    summary_file = os.path.join(eval_dir, "eval_summary.jsonl")
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(serializable_metrics, ensure_ascii=False) + "\n")
    print(f"[Eval/API] Saved eval metrics to {eval_file}")


def _compute_eval_metrics(episodes: list) -> dict:
    import numpy as np

    by_source_correct = defaultdict(list)
    by_source_pass = defaultdict(lambda: defaultdict(list))
    workflow_metrics_by_source = defaultdict(lambda: defaultdict(list))

    for episode in episodes:
        if episode is None:
            continue
        task = getattr(episode, "task", {}) or {}
        data_source = task.get("data_source") or "unknown"
        uid = str(getattr(episode, "id", ""))
        task_id = uid.rsplit(":", 1)[0] if uid else str(task.get("id", ""))
        is_correct = bool(getattr(episode, "is_correct", False))
        by_source_correct[data_source].append(is_correct)
        by_source_pass[data_source][task_id].append(is_correct)
        metrics = getattr(episode, "metrics", None) or {}
        for key, value in metrics.items():
            try:
                metric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(metric_value):
                workflow_metrics_by_source[data_source][key].append(metric_value)

    metrics = {}
    for data_source in sorted(by_source_correct):
        values = by_source_correct[data_source]
        pass_rates = by_source_pass[data_source]
        metrics[f"val/{data_source}/pass@1"] = float(np.mean(values)) if values else 0.0
        metrics[f"val/{data_source}/pass@{max((len(v) for v in pass_rates.values()), default=1)}"] = float(
            np.mean([1.0 if any(v) else 0.0 for v in pass_rates.values()])
        ) if pass_rates else 0.0
        for key, metric_values in workflow_metrics_by_source[data_source].items():
            if metric_values:
                metrics[f"val/{data_source}/{key}"] = float(np.mean(metric_values))
    return metrics


async def _run_api_model_eval_async(config, wf_args, rollout_save_dir: str, rollout_batch_size: int) -> dict:
    from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
    from rllm.engine.rollout.openai_engine import OpenAIEngine

    api_cfg = dict(_to_plain_container(wf_args.get("agent_model_api", {})) or {})
    if api_cfg.get("enable", False) is not True:
        raise ValueError("agent_model_api.enable must be true for API eval")

    # Validate before loading data or constructing any remote client.
    api_settings = resolve_agent_api(api_cfg)

    test_data = wf_args.get("test_data", None)
    test_files = _normalize_path_list(test_data, "test_data")
    if not test_files:
        raise ValueError("API eval requires test_data in config")
    tasks = _load_eval_tasks_from_files(test_files)

    n_val_samples = int(config.actor_rollout_ref.rollout.val_kwargs.get("n", 1))
    repeated_tasks = []
    task_ids = []
    for task in tasks:
        task_id = str(task.get("id") or len(task_ids))
        for _ in range(n_val_samples):
            repeated_tasks.append(task)
            task_ids.append(task_id)

    agent_config = dict(_to_plain_container(wf_args.get("agent_config", {})) or {})
    sampling_params = dict(api_cfg.get("sampling_params") or {})
    val_kwargs = config.actor_rollout_ref.rollout.val_kwargs
    sampling_params.setdefault("temperature", float(val_kwargs.get("temperature", 0.0)))
    sampling_params.setdefault("top_p", float(val_kwargs.get("top_p", 1.0)))
    if "presence_penalty" in val_kwargs:
        sampling_params.setdefault("presence_penalty", float(val_kwargs.get("presence_penalty", 0.0)))

    # vLLM/SGLang OpenAI-compatible servers accept non-standard knobs via
    # extra_body; the official OpenAI SDK rejects them as top-level kwargs.
    extra_body = dict(sampling_params.get("extra_body") or {})
    top_k = val_kwargs.get("top_k", None)
    if top_k is not None and int(top_k) >= 0:
        extra_body.setdefault("top_k", int(top_k))
    repetition_penalty = float(val_kwargs.get("repetition_penalty", 1.0))
    if repetition_penalty != 1.0:
        extra_body.setdefault("repetition_penalty", repetition_penalty)
    if extra_body:
        sampling_params["extra_body"] = extra_body

    rollout_engine = OpenAIEngine(
        model=api_settings.model,
        tokenizer=None,
        max_prompt_length=int(agent_config.get("max_prompt_tokens", 4096)),
        max_response_length=int(agent_config.get("max_response_tokens", 4096)),
        max_model_length=int(agent_config.get("max_prompt_tokens", 4096)) + int(agent_config.get("max_response_tokens", 4096)),
        api_retries=int(api_cfg.get("api_retries", api_cfg.get("max_retries", 3))),
        base_url=api_settings.base_url,
        api_key=api_settings.api_key,
        sampling_params=sampling_params,
    )
    rollout_engine.validate = True

    workflow_args = _build_common_workflow_args(config, wf_args, rollout_save_dir, rollout_batch_size)
    engine = AgentWorkflowEngine(
        workflow_cls=DeepResearchWorkflow,
        workflow_args=workflow_args,
        rollout_engine=rollout_engine,
        config=config,
        n_parallel_tasks=int(config.rllm.workflow.n_parallel_tasks),
        n_parallel_tools=int(config.rllm.workflow.get("n_parallel_tools", config.rllm.workflow.n_parallel_tasks * 8)),
        retry_limit=int(config.rllm.workflow.get("retry_limit", 3)),
    )
    engine.set_training_step(0, mode="val", epoch=0, total_steps=0)
    start = time.time()
    try:
        await engine.initialize_pool()
        print(
            f"[Eval/API] Running {len(repeated_tasks)} rollouts "
            f"({len(tasks)} samples x {n_val_samples}) with model={rollout_engine.model}"
        )
        episodes = await engine.execute_tasks(repeated_tasks, task_ids=task_ids)
    finally:
        engine.shutdown()
    metrics = _compute_eval_metrics(episodes)
    print(f"[Eval/API] Finished in {time.time() - start:.1f}s")
    return metrics


def _run_api_model_eval(config, wf_args, rollout_save_dir: str, rollout_batch_size: int) -> None:
    metrics = asyncio.run(
        _run_api_model_eval_async(config, wf_args, rollout_save_dir, rollout_batch_size)
    )
    display_metrics = _filter_eval_metrics_for_output(metrics)
    from pprint import pprint

    pprint(f"API validation metrics: {display_metrics}")
    _save_eval_metrics(
        config.trainer.get("default_local_dir", "checkpoints"), metrics, step=0
    )


def _build_dataset_register_name(train_descriptors: list[tuple[str, int]], test_descriptors: list[tuple[str, int]]) -> str:
    payload = {
        "train": sorted(train_descriptors),
        "test": sorted(test_descriptors),
    }
    parts = ["simple-agentic-search-data"]
    sorted_names = []
    for split_name, descriptors in (("train", payload["train"]), ("test", payload["test"])):
        if descriptors:
            total_count = sum(count for _, count in descriptors)
            parts.append(f"{split_name}{len(descriptors)}-{total_count}")
            sorted_names.extend(name for name, _ in descriptors)
    if sorted_names:
        parts.append("names-" + ".".join(sorted(sorted_names)))
    safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "-", "-".join(parts)).strip("-._")
    safe_base = (safe_base or "simple-agentic-search-data")[:160].rstrip("-._")
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    return f"{safe_base}-{digest}"


def load_datasets_from_config(config_or_path) -> tuple:
    """Load train/test datasets from a YAML config file or a dict.

    Accepts either:
        - A file path (str) to a YAML with train/test keys
        - A dict with train/test keys (lists of JSONL paths)

    Returns:
        (train_dataset, test_dataset)
    """
    if isinstance(config_or_path, (str, Path)):
        with open(config_or_path, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
    else:
        data_cfg = dict(config_or_path)

    train_files = _normalize_path_list(data_cfg.get("train", []), "train")
    test_files = _normalize_path_list(data_cfg.get("test", []), "test")

    if not train_files:
        raise ValueError("No train files specified")

    # Load and merge all train JSONL files
    all_train = []
    train_descriptors = []
    for path in train_files:
        samples = _load_jsonl(path)
        source_name = Path(path).stem
        train_descriptors.append((source_name, len(samples)))
        for idx, sample in enumerate(samples):
            sample.setdefault("data_source", source_name)
            sample.setdefault("data_source_path", path)
            sample["sample_idx"] = idx  # original line index for cache lookup
        logger.warning("Loaded %d samples from %s", len(samples), path)
        all_train.extend(samples)

    rng = random.Random(3407)
    rng.shuffle(all_train)
    logger.warning("Shuffled merged train samples with seed %d", 3407)

    train_formatted = _format_samples(all_train)
    logger.warning("Total train samples: %d", len(train_formatted))

    # Load and merge all test JSONL files
    all_test = []
    test_descriptors = []
    for path in test_files:
        samples = _load_jsonl(path)
        source_name = Path(path).stem
        test_descriptors.append((source_name, len(samples)))
        for idx, sample in enumerate(samples):
            sample.setdefault("data_source", source_name)
            sample.setdefault("data_source_path", path)
            sample["sample_idx"] = idx
        logger.warning("Loaded %d samples from %s", len(samples), path)
        all_test.extend(samples)

    test_formatted = _format_samples(all_test)
    logger.warning("Total test samples: %d", len(test_formatted))

    # Register to DatasetRegistry under a data-specific name so concurrent jobs
    # with different datasets cannot overwrite each other's parquet files.
    register_name = _build_dataset_register_name(train_descriptors, test_descriptors)
    logger.warning("Registering train/test dataset as %s", register_name)
    train_dataset = DatasetRegistry.register_dataset(register_name, train_formatted, "train")
    test_dataset = DatasetRegistry.register_dataset(register_name, test_formatted, "test")

    # Patch verl parquets to expose data_source, sample_idx, id as top-level columns
    _patch_verl_parquet(register_name, "train")
    _patch_verl_parquet(register_name, "test")

    return train_dataset, test_dataset


class _TeeStream:
    """Duplicate writes to both the original stream and a log file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data):
        self._original.write(data)
        try:
            self._log_file.write(data)
            self._log_file.flush()
        except Exception:
            pass

    def flush(self):
        self._original.flush()
        try:
            self._log_file.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._original, name)


def _setup_console_log(log_path: str):
    """Tee stdout and stderr to a log file so all print output is captured."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_fh = open(log_path, "a", encoding="utf-8")
    sys.stdout = _TeeStream(sys.stdout, log_fh)
    sys.stderr = _TeeStream(sys.stderr, log_fh)
    logger.warning("Console output is being logged to %s", log_path)


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer_megatron",
    version_base=None,
)
def main(config):
    # Set up console logging — capture all print/pprint output to file
    default_local_dir = config.trainer.get("default_local_dir", "checkpoints")
    console_log_path = os.path.join(default_local_dir, "console.log")
    _setup_console_log(console_log_path)

    wf_args = config.rllm.workflow.workflow_args

    # Hydra configures logging after module import; keep noisy tool cache-hit
    # diagnostics out of the terminal while preserving warnings and errors.
    logging.getLogger("simple_agentic_search.tools.region_search_tool").setLevel(logging.WARNING)

    # Determine rollout save directory
    default_local_dir = config.trainer.get("default_local_dir", "checkpoints")
    rollout_save_dir = os.path.join(default_local_dir, "rollout_data")
    rollout_save_dir = config.rllm.workflow.workflow_args.get(
        "rollout_save_dir", rollout_save_dir
    )

    # rollout_batch_size = train_batch_size * n_resp_per_prompt
    train_batch_size = config.data.get("train_batch_size", 16)
    n_resp_per_prompt = config.actor_rollout_ref.rollout.get("n", 1)
    rollout_batch_size = train_batch_size * n_resp_per_prompt

    agent_model_api = wf_args.get("agent_model_api", None)
    if agent_model_api is not None and bool(agent_model_api.get("enable", False)):
        if not config.trainer.get("val_only", False):
            raise ValueError("agent_model_api supports evaluation only; set trainer.val_only=true")
        _run_api_model_eval(config, wf_args, rollout_save_dir, rollout_batch_size)
        return

    # Load datasets: inline train_data/test_data lists > data_config file > registry
    train_data = wf_args.get("train_data", None)
    test_data = wf_args.get("test_data", None)

    if train_data is not None:
        # Inline data lists in experiment config. Normalize defensively because
        # Hydra command-line overrides can preserve a list as a string literal.
        train_dataset, test_dataset = load_datasets_from_config(
            {
                "train": _normalize_path_list(train_data, "train_data"),
                "test": _normalize_path_list(test_data, "test_data"),
            }
        )
    elif wf_args.get("data_config", None):
        # Separate data config YAML file (legacy)
        train_dataset, test_dataset = load_datasets_from_config(wf_args.data_config)
    else:
        # Fallback: load by registered name
        dataset_name = wf_args.get("dataset_name", "mars-agentic-search-data")
        train_dataset = DatasetRegistry.load_dataset(dataset_name, "train")
        test_dataset = DatasetRegistry.load_dataset(dataset_name, "test")

    workflow_args = _build_common_workflow_args(
        config, wf_args, rollout_save_dir, rollout_batch_size
    )

    from rllm.trainer.agent_trainer import AgentTrainer

    trainer = AgentTrainer(
        workflow_class=DeepResearchWorkflow,
        workflow_args=workflow_args,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
