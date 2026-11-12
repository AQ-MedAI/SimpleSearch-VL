"""Offline checks for release launchers; never contact services or print credentials.

Template checks need only PyYAML. Hydra is imported only to parse CLI overrides.
Runtime checks require explicit service configuration, not implicit provider fallbacks.
"""

import argparse
import copy
import json
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import yaml


class ConfigError(ValueError):
    """An actionable error safe to display without configuration values."""


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError("YAML mapping keys must be strings")
        if key in result:
            raise ConfigError(f"Duplicate YAML key at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_config(path):
    try:
        with open(path, encoding="utf-8") as stream:
            config = yaml.load(stream, Loader=_UniqueLoader)
    except (OSError, yaml.YAMLError):
        # YAML parser exceptions include source lines, which may contain secrets.
        raise ConfigError("Cannot read/parse configuration; check the file path and YAML syntax") from None
    if not isinstance(config, dict):
        raise ConfigError("Configuration must be a YAML mapping")
    reject_inline_secrets(config)
    return config


def reject_inline_secrets(value):
    """Forbid credential fields anywhere, including provider-specific headers."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower().replace("-", "_")
            secret = (
                name in {"api_key", "api_keys", "key", "token", "access_token", "auth_token", "authorization", "password", "secret"}
                or name.endswith(("_api_key", "_api_keys", "_access_key_id", "_access_key_secret", "_secret", "_password", "_token"))
            )
            if secret and child not in (None, ""):
                raise ConfigError("Inline credentials are forbidden in YAML/overrides; use the documented environment variables")
            reject_inline_secrets(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_inline_secrets(child)


def encode_hydra_value(value):
    """Hydra uses unquoted dict keys (unlike JSON), but quoted string values."""
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key) for key in value):
            raise ConfigError("API option mapping keys must be identifiers (letters/digits/underscore/hyphen)")
        return "{" + ",".join(f"{key}:{encode_hydra_value(child)}" for key, child in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(encode_hydra_value(child) for child in value) + "]"
    try:
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError):
        raise ConfigError("API options must contain finite JSON-compatible values") from None


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _text(value, name, *, allow_blank=False):
    if not isinstance(value, str) or (not allow_blank and not value.strip()):
        raise ConfigError(f"{name} must be a nonempty string" if not allow_blank else f"{name} must be a string")
    if value != value.strip() or any(c in value for c in "\r\n\x00"):
        raise ConfigError(f"{name} must not contain surrounding whitespace or control characters")
    if "${" in value or (value.startswith("<") and value.endswith(">")):
        raise ConfigError(f"{name} must be filled explicitly, not with an interpolation or placeholder")
    return value


def _url(value, name, *, allow_blank=False):
    text = _text(value, name, allow_blank=allow_blank)
    if not text and allow_blank:
        return text
    try:
        parsed = urlsplit(text)
        _ = parsed.port
        valid = (
            parsed.scheme in {"http", "https"} and parsed.hostname
            and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment
            and not any(c.isspace() for c in text)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ConfigError(f"{name} must be an http(s) URL without embedded credentials, query, or fragment")
    return text


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class AgentAPISettings:
    model: str
    base_url: str
    api_key: str = field(repr=False)


def resolve_agent_api(config, *, environ=None, template=False):
    """Explicit endpoint/model, isolated env key; never fall back to OPENAI_API_KEY.

    Empty string (not None) intentionally disables SDK environment-key fallback
    for endpoints explicitly marked allow_unauthenticated.
    """
    api = _mapping(config, "agent_model_api")
    reject_inline_secrets(api)
    model = _text(api.get("model", ""), "agent_model_api.model", allow_blank=template)
    base_url = _url(api.get("base_url", ""), "agent_model_api.base_url", allow_blank=template)
    env_name = api.get("api_key_env", "AGENT_API_KEY")
    if not isinstance(env_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
        raise ConfigError("agent_model_api.api_key_env must name an environment variable, not contain a key")
    unauthenticated = api.get("allow_unauthenticated", False)
    if type(unauthenticated) is not bool:
        raise ConfigError("agent_model_api.allow_unauthenticated must be true or false")
    _positive_int(api.get("api_retries", 3), "agent_model_api.api_retries")
    _mapping(api.get("sampling_params", {}), "agent_model_api.sampling_params")
    encode_hydra_value(api)  # Validate options before the shell serializes them.
    key = ""
    if not template and not unauthenticated:
        env = os.environ if environ is None else environ
        key = env.get(env_name, "")
        if not key or not key.strip() or key != key.strip():
            raise ConfigError("Agent API credential is missing/invalid; set the variable named by agent_model_api.api_key_env")
    return AgentAPISettings(model=model, base_url=base_url, api_key=key)


def validate_services(config, environ=None):
    env = os.environ if environ is None else environ
    # train.py always supplies a judge callback. Even removing the accuracy
    # reward (or setting its weight to zero) does not disable judge calls.
    required = ["JUDGE_MODELS", "JUDGE_BASE_URLS", "JUDGE_API_KEYS"]
    tools = config.get("tools_config", {}).get("enabled_tools", [])
    if "text_search" in tools or "image_search" in tools:
        required.append("SERP_API_KEY")
    if "image_search" in tools:
        required.extend(["OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_ENDPOINT", "OSS_BUCKET_NAME"])
    if "visit" in tools and (env.get("EXTRACT_MODEL") or "self").strip().lower() != "self":
        required.extend(["EXTRACT_MODEL", "EXTRACT_URL"])
    missing = [name for name in required if not env.get(name, "").strip()]
    if missing:
        raise ConfigError("Missing runtime environment variables: " + ", ".join(missing))
    if "JUDGE_MODELS" in required:
        groups = [env[name].split(",") for name in ("JUDGE_MODELS", "JUDGE_BASE_URLS", "JUDGE_API_KEYS")]
        if len({len(group) for group in groups}) != 1 or any(not item.strip() for group in groups for item in group):
            raise ConfigError("JUDGE_MODELS, JUDGE_BASE_URLS, JUDGE_API_KEYS must have matching nonempty comma-separated entries")
        for url in groups[1]:
            _url(url.strip(), "JUDGE_BASE_URLS entry")
        # This legacy setting takes precedence over API configs in the reward code.
        if env.get("JUDGE_URL", "").strip():
            raise ConfigError("Unset legacy JUDGE_URL when using the release JUDGE_BASE_URLS configuration")
    for name in ("OSS_ENDPOINT", "TEXT_SEARCH_URL", "READER_URL"):
        if env.get(name, "").strip():
            _url(env[name], name)
    if "EXTRACT_URL" in required:
        for url in env["EXTRACT_URL"].split(","):
            _url(url.strip(), "EXTRACT_URL entry")


def validate_config(config, *, template=False, environ=None):
    reject_inline_secrets(config)
    mode = config.get("mode")
    if mode not in ("train", "eval"):
        raise ConfigError("mode must be train or eval")
    api = _mapping(config.get("agent_model_api", {}), "agent_model_api")
    enabled = api.get("enable", False)
    if type(enabled) is not bool:
        raise ConfigError("agent_model_api.enable must be true or false")
    if enabled:
        if mode != "eval":
            raise ConfigError("agent_model_api supports mode=eval only")
        resolve_agent_api(api, environ=environ, template=template)
    else:
        model = _text(config.get("model_path", ""), "model_path", allow_blank=template)
        if not template and not Path(model).is_dir():
            # Local paths must exist. Remote model IDs are not downloaded by this check.
            if model.startswith((".", "/", "~")) or not re.fullmatch(r"[\w.-]+/[\w.-]+", model):
                raise ConfigError("model_path must be an existing checkpoint directory or an explicit namespace/model ID")
    for field_name in ("train_data", "test_data"):
        paths = config.get(field_name, [])
        if not isinstance(paths, list) or (not paths and not (enabled and field_name == "train_data")):
            raise ConfigError(f"{field_name} must be a nonempty list (only API eval may omit train_data)")
        for path in paths:
            _text(path, f"{field_name} path")
            if not template and (not Path(path).is_file() or Path(path).stat().st_size == 0):
                raise ConfigError(f"{field_name} contains a missing/empty file; paths are relative to the rllm directory")
    for name in ("n_parallel_tasks", "n_parallel_tools", "val_batch_size"):
        _positive_int(config.get(name), name)
    agent = _mapping(config.get("agent_config", {}), "agent_config")
    run_timeout = _positive_int(agent.get("agent_run_timeout"), "agent_config.agent_run_timeout")
    final_timeout = _positive_int(agent.get("force_final_answer_timeout"), "agent_config.force_final_answer_timeout")
    if _positive_int(config.get("workflow_timeout"), "workflow_timeout") <= run_timeout + final_timeout:
        raise ConfigError("workflow_timeout must exceed agent_run_timeout + force_final_answer_timeout (leave time for rewards)")
    tools = _mapping(config.get("tools_config", {}), "tools_config")
    names = tools.get("enabled_tools")
    if not isinstance(names, list) or not names or any(not isinstance(name, str) or name not in {"text_search", "image_search", "visit", "zoom_in"} for name in names):
        raise ConfigError("tools_config.enabled_tools must list supported tools explicitly")
    _mapping(config.get("reward", {}), "reward")
    if not template:
        validate_services(config, environ)


def apply_overrides(config, overrides):
    """Apply preflight-relevant Hydra overrides without importing the training stack."""
    result = copy.deepcopy(config)
    if not overrides:
        return result
    from hydra.core.override_parser.overrides_parser import OverridesParser

    try:
        parsed = OverridesParser.create().parse_overrides(overrides)
    except Exception:
        raise ConfigError("Cannot parse Hydra overrides; check their syntax (values omitted for safety)") from None
    mapping = {
        "actor_rollout_ref.model.path": "model_path",
        "data.val_batch_size": "val_batch_size",
        "rllm.workflow.n_parallel_tasks": "n_parallel_tasks",
        "rllm.workflow.n_parallel_tools": "n_parallel_tools",
    }
    prefix = "rllm.workflow.workflow_args."
    for override in parsed:
        if override.is_sweep_override() or override.is_delete():
            raise ConfigError("Release preflight accepts single-value overrides only, not sweeps/deletions")
        key, value = override.key_or_group, override.value()
        reject_inline_secrets({key.rsplit(".", 1)[-1]: value})
        if key == "trainer.val_only":
            result["mode"] = "eval" if value else "train"
            continue
        target = mapping.get(key)
        if key.startswith(prefix):
            target = key[len(prefix):]
            if target == "timeout":
                target = "workflow_timeout"
        if target:
            parts = target.split(".")
            node = result
            for part in parts[:-1]:
                node = node.setdefault(part, {})
                if not isinstance(node, dict):
                    raise ConfigError("Override parent must be a mapping")
            node[parts[-1]] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", action="store_true", help="Check public templates without credentials, data, or weights")
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    try:
        config = apply_overrides(load_config(args.config), args.overrides)
        validate_config(config, template=args.template)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    print("Configuration check passed (offline; no services contacted).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
