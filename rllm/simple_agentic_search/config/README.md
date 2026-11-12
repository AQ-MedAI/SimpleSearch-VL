# Public experiment templates

These templates contain no credentials. They are starting points, not turnkey
runs or hardware-independent reproductions. See the root README for installation,
datasets, checkpoints, and service setup.

| Template | Purpose | Required user changes |
| --- | --- | --- |
| `eval_8B.yaml` | Local 8B RL-checkpoint evaluation | Download weights/data or change paths; fill service environment. |
| `eval_30B.yaml` | Local 30B-A3B RL-checkpoint evaluation | Same; review the 8-GPU TP/EP/offload profile. |
| `eval_apimodel.yaml` | Remote API evaluation, no Ray or local weights | Fill `agent_model_api.base_url` and `model`; set `AGENT_API_KEY` in the environment. |
| `qwen3vl8b_sft_dapo_2x_dynroll_4_6_150step.yaml` | 8B RL with FAR | Set the blank `model_path` to your own SFT checkpoint (or original model); fill service environment. |
| `qwen3vl30b_sft_dapo_2x_dynroll_4_6_150step.yaml` | 30B-A3B RL with FAR | Same; review the 8-GPU MoE/offload profile. |

## Local customization

Run the following from the repository's `rllm/` directory:

```bash
mkdir -p simple_agentic_search/config/local
cp simple_agentic_search/config/eval_apimodel.yaml simple_agentic_search/config/local/eval_api.yaml
# Edit the new YAML and your private rllm/.env first.
bash simple_agentic_search/run/run.sh --check simple_agentic_search/config/local/eval_api.yaml
bash simple_agentic_search/run/run.sh simple_agentic_search/config/local/eval_api.yaml
```

- `config/local/` and `*.local.yaml` / `*.local.yml` are ignored by Git. Keep the
  public templates blank where user input is required.
- Model, data, cache and output paths resolve relative to **`rllm/`**, including
  configs stored in `config/local/`. Config-file arguments themselves resolve
  relative to the caller's working directory.
- Never fill `agent_model_api.api_key` in YAML. Leave it `""` and set the variable
  named by `api_key_env` (default `AGENT_API_KEY`) in the local environment. Inline
  credential fields are rejected; snapshots/Hydra output must not contain keys.
- `base_url` and `model` have no implicit provider/model fallback. Only set
  `allow_unauthenticated: true` for a deliberately unauthenticated endpoint; this
  bypasses the key environment variable rather than reusing an unrelated key.
- API evaluation requires only `test_data`. Local-checkpoint evaluation also
  loads `train_data` because it uses the shared trainer.
- Search and accuracy-judge credentials are separate from the agent API key.
  `image_search` also needs OSS on cache misses. Refer to `rllm/.env.example`.

## Defaults and reproducibility

- All benchmark paths use `../data/SimpleSearch-VL/Evaluation/`. Training uses
  full validation JSONLs rather than assuming unpublished `*_100.jsonl` files.
  For cheaper validation, explicitly create/select and record local subsets.
- Training preserves FAR/DAPO: 150 steps, 2x prompt expansion, 4–6 rollouts,
  expansion of all-wrong groups, and full-rollout advantage computation.
- Eval-only configs disable training-only rollout/advantage/efficiency features
  and use console logging. Their optimizer/parallel fields are shared-schema
  settings, not an instruction to train. API evaluation ignores local GPU fields.
- Eval workflow hard timeouts are 1000 seconds for local checkpoints and 1200
  seconds for API models. These include reward computation, not just the agent;
  raise them deliberately for slower services. Timeouts can change completion rates.
- All profiles share `simple_agentic_search/cache/mars_cache.db` and enable direct
  online thumbnail downloads on thumbnail-cache misses. The defaults assume
  outbound network access from every tool-executing worker, not a relay service.
  An empty cache can be used with the required services configured and reachable,
  but incurs API calls. See the network settings below before disabling downloads.
- Local training/eval profiles assume eight GPUs per node. API evaluation starts
  at four concurrent tasks; adjust concurrency and sampling for provider limits.
- Validation subsets, timeouts, cache contents, online search changes and provider
  settings affect measured results. These portable defaults are not a claim of
  bit-for-bit reproduction of a historical run.

## Thumbnail downloads and network access

All five public templates assume that every process/worker running `image_search`
can reach the returned image URLs directly over HTTP(S). No thumbnail relay is
required. In multi-node runs, access is needed on the workers, not just the head
node. Search/OSS service configuration is still required separately.

Settings below live under `tools_config.image_search`:

| Setting | Public default | Meaning |
| --- | --- | --- |
| `use_thumbnails` | `true` | Attach available search-result thumbnails as visual evidence. `false` skips fetching and attaching thumbnails. |
| `thumbnail_online_download` | `true` | Read cached thumbnail bytes first, then directly download missing ones. `false` permits cached thumbnails only; it does not disable online image search or other tools. |
| `thumbnail_download_timeout` | `10.0` | HTTP client network timeout in seconds, not a total workflow deadline. |
| `thumbnail_max_bytes` | `5242880` | Reject images larger than 5 MiB **after** downloading the response; this is not a streaming/bandwidth cap. |
| `thumbnail_trust_env` | `false` | Do not inherit environment proxy settings for thumbnail requests. Set `true` only when intentionally using your environment's proxy/certificate settings. |

Region-search results and thumbnail bytes are cached separately: a cached search
result may still require an image download. Missing, failed, or rejected thumbnails
are omitted while textual search results remain available. This reduces visual
evidence and can affect training/evaluation results; it is not equivalent to a run
with all thumbnails available.

For restricted networks, a proxy/relay is an optional deployment workaround, not
an open-source prerequisite. To use an environment-configured proxy, set
`thumbnail_trust_env: true` in your private YAML and configure `HTTP_PROXY`,
`HTTPS_PROXY` (or `ALL_PROXY`, with `NO_PROXY` as needed) on every worker. Keep
proxy credentials in your private environment, not the public YAML. No relay is
automatically selected or configured. This flag affects only the thumbnail HTTP
client; it does not configure all search, reader, or model API traffic. If downloads
must be disabled, set `thumbnail_online_download: false` and provide a populated
thumbnail cache or accept missing uncached visual evidence.

## Offline checks

```bash
# Template-only: no local credentials, weights or datasets required.
for cfg in simple_agentic_search/config/*.yaml; do
  python -m simple_agentic_search.config_validation --template "$cfg" || break
done

# Real launch preflight: loads .env; checks required fields/files/env, not services.
bash simple_agentic_search/run/run.sh --check simple_agentic_search/config/local/eval_api.yaml
```

Preflight never verifies key validity, remote access, image paths, dataset schema,
model architecture, GPU memory or end-to-end correctness. It imports no GPU stack.
Normal launchers run preflight before snapshots or Ray setup. Keep Ray launches on
dedicated nodes: `run_multi_node.sh` stops existing Ray processes after preflight.
