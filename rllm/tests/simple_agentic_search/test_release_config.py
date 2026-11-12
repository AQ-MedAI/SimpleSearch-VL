"""Release checks use temporary files/dummy credentials; never call Ray or services."""

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from simple_agentic_search.config_validation import (
    ConfigError,
    apply_overrides,
    load_config,
    resolve_agent_api,
    validate_config,
    validate_services,
)

ROOT = Path(__file__).resolve().parents[3]
RLLM = ROOT / "rllm"
CONFIGS = RLLM / "simple_agentic_search/config"
RUN = RLLM / "simple_agentic_search/run"
TEST_KEY = "unit-test-credential-not-a-real-key"


def runtime_env():
    return {
        "AGENT_API_KEY": TEST_KEY,
        "JUDGE_MODELS": "judge-model",
        "JUDGE_BASE_URLS": "http://127.0.0.1:9001/v1",
        "JUDGE_API_KEYS": "unused",
        "SERP_API_KEY": "unused",
        "OSS_ACCESS_KEY_ID": "unused",
        "OSS_ACCESS_KEY_SECRET": "unused",
        "OSS_ENDPOINT": "https://storage.example.invalid",
        "OSS_BUCKET_NAME": "test-only",
        "EXTRACT_MODEL": "self",
    }


def api_config(tmp_path):
    config = load_config(CONFIGS / "eval_apimodel.yaml")
    config["agent_model_api"].update(base_url="http://127.0.0.1:9000/v1", model="test-model")
    data = tmp_path / "data with spaces.jsonl"
    data.write_text('{}\n')
    config["test_data"] = [str(data)]
    config["default_local_dir"] = str(tmp_path / "artifacts")
    return config


def subprocess_env(tmp_path):
    # Deliberately do not inherit any real API keys or load the user's .env.
    return {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "SIMPLESEARCH_ENV_FILE": "",
    }


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_public_templates(path):
    config = load_config(path)
    validate_config(config, template=True, environ={})
    assert all(p.startswith("../data/SimpleSearch-VL/Evaluation/") for p in config["test_data"])
    assert len(config["test_data"]) == 6
    assert not any("_100.jsonl" in p for p in config["test_data"])
    assert config["tools_config"]["cache_db_path"] == "simple_agentic_search/cache/mars_cache.db"
    assert config["tools_config"]["image_search"]["thumbnail_online_download"] is True
    if config["mode"] == "train":
        assert config["model_path"] == ""
        assert config["total_training_steps"] == 150
        assert config["dynamic_rollout"]["enable"] is True
        assert config["dynamic_rollout"]["min_rollouts"] == 4
        assert config["dynamic_rollout"]["max_rollouts"] == 6
        assert config["dynamic_rollout"]["use_full_rollouts_for_advantage"] is True
        assert config["dapo_sampling"]["multiplier"] == 2.0
    else:
        for name in ("dynamic_rollout", "dapo_sampling", "stepwise_advantage"):
            assert config[name]["enable"] is False
        assert config["reward"]["global_efficiency"]["enable"] is False


def test_api_template_is_intentionally_unconfigured():
    config = load_config(CONFIGS / "eval_apimodel.yaml")
    api = config["agent_model_api"]
    assert api["api_key"] == api["base_url"] == api["model"] == ""
    assert api["api_key_env"] == "AGENT_API_KEY"
    assert api["sampling_params"] == {}
    assert config["train_data"] == []
    with pytest.raises(ConfigError, match="model"):
        validate_config(config, environ={})


def test_credentials_are_explicit_and_not_in_repr(tmp_path):
    api = api_config(tmp_path)["agent_model_api"]
    with pytest.raises(ConfigError, match="credential"):
        resolve_agent_api(api, environ={"OPENAI_API_KEY": TEST_KEY})
    settings = resolve_agent_api(api, environ={"AGENT_API_KEY": TEST_KEY})
    assert settings.api_key == TEST_KEY
    assert TEST_KEY not in repr(settings)
    api["allow_unauthenticated"] = True
    settings = resolve_agent_api(api, environ={"AGENT_API_KEY": TEST_KEY, "OPENAI_API_KEY": TEST_KEY})
    assert settings.api_key == ""  # Not None: no SDK fallback to an unrelated key.


@pytest.mark.parametrize("field,value", [
    ("api_key", TEST_KEY),
    ("api_key", "${oc.env:AGENT_API_KEY}"),
    ("base_url", "https://user:secret@example.invalid/v1"),
    ("base_url", "https://example.invalid/v1?api_key=secret"),
    ("base_url", ""),
    ("model", ""),
    ("allow_unauthenticated", "false"),
    ("api_retries", 0),
    ("api_key_env", "not a variable"),
    ("sampling_params", {"extra_headers": {"Authorization": TEST_KEY}}),
])
def test_invalid_api_settings_are_rejected_without_echoing_values(tmp_path, field, value):
    api = api_config(tmp_path)["agent_model_api"]
    api[field] = value
    with pytest.raises(ConfigError) as exc:
        resolve_agent_api(api, environ=runtime_env())
    assert TEST_KEY not in str(exc.value)
    assert "user:secret" not in str(exc.value)


def test_duplicate_and_malformed_yaml_errors_do_not_echo_source(tmp_path):
    path = tmp_path / "bad.yaml"
    for content in (f"model: a\nmodel: {TEST_KEY}\n", f"api_key: [{TEST_KEY}\n"):
        path.write_text(content)
        with pytest.raises(ConfigError) as exc:
            load_config(path)
        assert TEST_KEY not in str(exc.value)


def test_runtime_paths_and_environment(tmp_path):
    config = api_config(tmp_path)
    validate_config(config, environ=runtime_env())
    env = runtime_env()
    del env["OSS_ACCESS_KEY_SECRET"]
    with pytest.raises(ConfigError, match="OSS_ACCESS_KEY_SECRET"):
        validate_config(config, environ=env)
    env = runtime_env()
    env["JUDGE_MODELS"] = "model-a,model-b"
    with pytest.raises(ConfigError, match="matching nonempty"):
        validate_services(config, env)
    config["test_data"] = [str(tmp_path / "missing.jsonl")]
    with pytest.raises(ConfigError, match="missing/empty file"):
        validate_config(config, environ=runtime_env())


def test_validation_follows_enabled_tools_but_keeps_legacy_judge_required(tmp_path):
    config = api_config(tmp_path)
    config["tools_config"]["enabled_tools"] = ["zoom_in"]
    config["reward"] = {"format": 0.5}
    env = {key: value for key, value in runtime_env().items() if key.startswith("JUDGE_") or key == "AGENT_API_KEY"}
    validate_config(config, environ=env)
    with pytest.raises(ConfigError, match="JUDGE_MODELS"):
        validate_config(config, environ={"AGENT_API_KEY": TEST_KEY})
    config["reward"]["accuracy"] = 0.0
    with pytest.raises(ConfigError, match="JUDGE_MODELS"):
        validate_config(config, environ={"AGENT_API_KEY": TEST_KEY})


def test_overrides_are_validated_instead_of_ignoring_them(tmp_path):
    config = api_config(tmp_path)
    original = copy.deepcopy(config)
    updated = apply_overrides(config, [
        '++rllm.workflow.workflow_args.agent_model_api.model="different model"',
        '++rllm.workflow.workflow_args.agent_model_api.allow_unauthenticated=true',
        'rllm.workflow.n_parallel_tasks=2',
    ])
    validate_config(updated, environ={k: v for k, v in runtime_env().items() if k != "AGENT_API_KEY"})
    assert updated["agent_model_api"]["model"] == "different model"
    assert config == original
    for override in (
        f"++rllm.workflow.workflow_args.agent_model_api.api_key={TEST_KEY}",
        f"++rllm.workflow.workflow_args.agent_model_api={{api_key:{TEST_KEY}}}",
    ):
        with pytest.raises(ConfigError) as exc:
            apply_overrides(config, [override])
        assert TEST_KEY not in str(exc.value)
    updated = apply_overrides(config, ["trainer.val_only=false"])
    with pytest.raises(ConfigError, match="eval only"):
        validate_config(updated, environ=runtime_env())


def test_public_env_is_sourceable_and_credentials_are_blank(tmp_path):
    result = subprocess.run(
        ["bash", "-c", 'set -euo pipefail; source "$1"; for name in AGENT_API_KEY JUDGE_API_KEYS SERP_API_KEY JINA_API_KEY EXTRACT_KEY OSS_ACCESS_KEY_ID OSS_ACCESS_KEY_SECRET; do test -z "${!name}"; done', "bash", str(RLLM / ".env.example")],
        env=subprocess_env(tmp_path), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("launcher", ["run.sh", "run_multi_node.sh"])
def test_check_only_has_no_runtime_side_effects_and_hides_sourced_keys(tmp_path, launcher):
    config = api_config(tmp_path)
    config_path = tmp_path / "config with 'quote'.yaml"
    config_path.write_text(yaml.safe_dump(config))
    private_env = tmp_path / ".env"
    private_env.write_text("\n".join(f'{key}="{value}"' for key, value in runtime_env().items()))
    env = subprocess_env(tmp_path)
    env["SIMPLESEARCH_ENV_FILE"] = str(private_env)
    result = subprocess.run(
        ["bash", "-x", str(RUN / launcher), "--check", str(config_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert TEST_KEY not in result.stdout + result.stderr
    assert "check passed" in result.stdout
    assert not (tmp_path / "artifacts").exists()


def test_invalid_config_fails_before_ray_is_touched(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    marker = tmp_path / "ray-was-called"
    ray = fake_bin / "ray"
    ray.write_text(f'#!/bin/sh\ntouch "{marker}"\nexit 99\n')
    ray.chmod(0o755)
    env = subprocess_env(tmp_path)
    env["PATH"] = str(fake_bin) + ":" + env["PATH"]
    result = subprocess.run(
        ["bash", str(RUN / "run_multi_node.sh"), str(CONFIGS / "eval_apimodel.yaml")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert not marker.exists()


@pytest.mark.parametrize("template_path", sorted(CONFIGS.glob("*.yaml")), ids=lambda p: p.name)
def test_launch_preserves_override_boundaries_and_keeps_key_out_of_argv(tmp_path, template_path):
    """Run the real shell assembly, but replace the final Python/Ray/rsync commands."""
    is_api = template_path.name == "eval_apimodel.yaml"
    if is_api:
        config = api_config(tmp_path)
        config["agent_model_api"]["model"] = "model with spaces"
        config["agent_model_api"]["sampling_params"] = {"extra_body": {"option": "value with spaces"}}
    else:
        config = load_config(template_path)
        data = tmp_path / "data with spaces.jsonl"
        data.write_text('{}\n')
        config["train_data"] = config["test_data"] = [str(data)]
        model = tmp_path / "model with spaces"
        model.mkdir()
        config["model_path"] = str(model)
        config["default_local_dir"] = str(tmp_path / "artifacts")
    config_path = tmp_path / "config with 'quote'.yaml"
    config_path.write_text(yaml.safe_dump(config))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    output = tmp_path / "argv.json"
    scripts = {
        "python": f'#!/usr/bin/env python3\nimport json,sys\nfrom pathlib import Path\nPath({str(output)!r}).write_text(json.dumps(sys.argv[1:]))\n',
        "rsync": "#!/bin/sh\nexit 0\n",
        "ray": "#!/bin/sh\nexit 99\n",
    }
    for name, content in scripts.items():
        path = fake_bin / name
        path.write_text(content)
        path.chmod(0o755)
    env = subprocess_env(tmp_path)
    env.update(runtime_env())
    env["PATH"] = str(fake_bin) + ":" + env["PATH"]
    result = subprocess.run(
        ["bash", str(RUN / ("run_multi_node.sh" if is_api else "run.sh")), str(config_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert TEST_KEY not in result.stdout + result.stderr + output.read_text()
    args = json.loads(output.read_text())
    assert args[:2] == ["-m", "simple_agentic_search.train"]
    from hydra.core.override_parser.overrides_parser import OverridesParser

    values = {item.key_or_group: item.value() for item in OverridesParser.create().parse_overrides(args[2:])}
    prefix = "rllm.workflow.workflow_args."
    if is_api:
        assert values[prefix + "agent_model_api.model"] == "model with spaces"
        assert values[prefix + "agent_model_api.sampling_params"]["extra_body"]["option"] == "value with spaces"
    else:
        assert values["actor_rollout_ref.model.path"] == config["model_path"]
    assert values[prefix + "test_data"] == config["test_data"]
    assert prefix + "agent_model_api.api_key" not in values
    assert values.get("trainer.val_only", False) == (config["mode"] == "eval")
    # Compose the real trainer schema without importing GPU packages through
    # its normal pkg:// search path. Only the source-location syntax differs.
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(RLLM / "rllm/trainer/config"), version_base=None):
        composed = compose(
            config_name="agent_ppo_trainer_megatron",
            overrides=[*args[2:], f"hydra.searchpath=[file://{RLLM / 'verl/verl/trainer/config'}]"],
        )
    if is_api:
        assert composed.rllm.workflow.workflow_args.agent_model_api.model == "model with spaces"
        assert composed.trainer.n_gpus_per_node == 0
    else:
        assert composed.actor_rollout_ref.model.path == config["model_path"]
        assert composed.trainer.n_gpus_per_node == 8
        if config["mode"] == "train":
            assert composed.trainer.total_training_steps == 150
            assert composed.rllm.dynamic_rollout.min_rollouts == 4
            assert composed.rllm.dynamic_rollout.max_rollouts == 6
            assert composed.rllm.dapo_sampling.multiplier == 2.0


def test_private_files_are_ignored_but_templates_are_not():
    private = [
        "rllm/.env", "rllm/.env.production", "rllm/simple_agentic_search/config/local/run.yaml",
        "rllm/simple_agentic_search/config/my.local.yaml", "my_script/private.py", "docs/private.md",
        "rllm/simple_agentic_search/cache/mars_cache.db-wal", "data/private.jsonl",
    ]
    for path in private:
        assert subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT).returncode == 0
    for path in ["rllm/.env.example", "rllm/simple_agentic_search/config/eval_apimodel.yaml"]:
        assert subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT).returncode == 1
