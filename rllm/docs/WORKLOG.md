## 2026-05-20 - Config-driven image-search-first rollout step

- Added and wired `agent_config.image_search_first` through `run.sh` to let the agent prepend a scripted whole-image `image_search` call before the first model response when input images are available.
- The scripted step writes a simple `<thinking>` message plus strict JSON tool call over all input `img_idx` values with `[0, 0, 1000, 1000]` boxes, then resumes the normal model/tool loop.
- Enabled the switch in the MC-Search multiframe Step2 probe/full configs.
- Verified with `py_compile`, `bash -n`, a fake two-image smoke test, and a real two-sample rollout smoke that the first tool call is generated before the first model response.

## 2026-05-14 - Align rollout progress signal with reward signal

- Changed early-stop progress signal counting to use trajectory reward when available, matching DAPO/dynamic stop-condition signal detection.
- This avoids confusing logs where the progress bar shows fewer binary correctness signals than the DAPO reward-signal stop condition.

## 2026-05-14 - Refresh global efficiency rollout visualization

- Updated single-step rollout dashboards to present workflow-level global efficiency with the new step-end state update semantics.
- Replaced the stale negative-bonus emphasis with history-reference coverage, positive bonus counts, round/tool bonus source counts, and derived state updates after step selection.
- Extended all-step dashboards with reference coverage and state-update trends so the plots distinguish bonus computation from persistent state mutation.

## 2026-05-14 - Defer global efficiency state updates to step end

- Changed workflow-level global efficiency reward computation to read historical references without mutating state per rollout.
- Added a trainer step-end update that writes successful retained episodes to `global_efficiency_state.json` after rollout selection/filtering is complete.
- Verified that same-step rollouts do not affect each other's reference and that later steps receive positive-only round/tool bonuses from the updated state.

## 2026-05-14 - Move global efficiency reward into workflow

- Removed the trainer-side round/group efficiency postprocessing path and deleted the old `GroupEfficiencyReward` helper.
- Added workflow-level persistent `GlobalEfficiencyReward` with file-locked per-sample state, positive-only round/tool bonuses, and no `correct_only` config knob.
- Updated `run.sh` to inject `global_efficiency.state_path` from the checkpoint directory when the reward is enabled.
- Added `mars_agentic_search/config/ab_sota/dapo_dynamic_global_efficiency_base.yaml` as a DAPO + dynamic rollout base config.
- Verified with `py_compile`, a global-efficiency smoke test, and YAML parsing.

## 2026-05-14 - Disable thinking mode for vLLM judge

- Added `chat_template_kwargs={"enable_thinking": False}` to the vLLM judge request payload in `rllm.rewards.deepresearch_reward`, so Qwen3.6 writes `correct: yes/no` to `message.content` instead of spending tokens in `message.reasoning`.
- Verified the real `RewardDeepResearchFn` path against `JUDGE_MODEL=Qwen3.6-35B-A3B` / `JUDGE_URL=http://200.20.7.6:8003`: correct entity, wrong entity, and placeholder cases all returned in 0.22-0.33s with `judge_decided=True`.

## 2026-05-14 - Enforce tool-call arguments object in format reward

- Tightened `mars_agentic_search.reward.format_reward` so `<tool_call>` JSON with non-object `arguments` is a structural format error. This catches malformed calls such as `{"name":"text_search","arguments":[...]}` that previously bypassed format reward and failed only at tool execution time.
- Added an execution-side guard in `mars_agentic_search.deepresearch_agent` so malformed non-object tool arguments return a clear `[Tool Format Error]` instead of reaching `tool.call(**args)`.
- Added `tests/rewards/test_format_reward.py` coverage for valid object arguments and invalid list arguments.
- Verified with `PYTHONPATH=/personal/personal_code/OpenMMSeeker/rllm /root/miniconda3/envs/vision_dr/bin/python -m pytest tests/rewards/test_format_reward.py tests/rewards/test_deepresearch_reward.py -q` (`7 passed`).


## 2026-05-07 - Workflow hard timeout for Mars rollouts

- Exposed top-level `workflow_timeout` in `mars_agentic_search/run/run.sh`, defaulting to `agent_config.agent_run_timeout` for backward compatibility.
- Passed `rllm.workflow.workflow_args.timeout` into `DeepResearchWorkflow` through `mars_agentic_search/train.py`, along with the existing `agent_config`.
- Added rollout backend `abort_all_requests()` cleanup when base `Workflow.run_with_termination_handling()` hits `asyncio.TimeoutError`, so workflow-level timeout can cancel in-flight LLM backend requests instead of only checking at the next agent loop turn.
- Verified with `python -m py_compile mars_agentic_search/train.py rllm/workflows/workflow.py` and `bash -n mars_agentic_search/run/run.sh`.
## 2026-05-07 - M3VQA single-answer RL filtering pipeline

- Updated `data/VDR-data/train/M3VQA/1_preprocess.py` so the default main pool keeps only single-answer samples; multi-answer samples require `--allow-multi-answer`.
- Pointed `2_generate_answers.py` at the remote Qwen3-VL-8B vLLM endpoint `http://200.20.7.6:8003/v1`.
- Reworked `3_judge.py` prompt for single primary answer plus aliases from `answer_evals[0]`.
- Rewrote `4_select.py` to select hard single-answer samples with a soft 1k-2k target and diversity across hop, question type, and entity groups.
- Refreshed `FILTER_PIPELINE.md` with the new rerun sequence and expected single-answer Step 1 stats.
- Verified with `python -m py_compile 1_preprocess.py 2_generate_answers.py 3_judge.py 4_select.py` and a read-only Step 1 stats check (`13125 -> 4884`, all single-answer).
## 2026-05-07 - Reuse 8B vLLM for M3VQA judge

- Changed `data/VDR-data/train/M3VQA/3_judge.py` from the AntChat Qwen3-235B endpoint to the local OpenAI-compatible vLLM endpoint `http://200.20.7.6:8003/v1`.
- Default judge model is now `Qwen3-VL-8B-Instruct`, matching Step 2, with `--judge-url` and `--judge-model` overrides.
- Removed hardcoded AntChat API-key usage from the judge path.
- Verified with `python -m py_compile 3_judge.py` and a one-sample non-writing smoke test (`Caribbean` vs `North America` -> `no`).
## 2026-05-08 - M3VQA Step5 reliability audit and relaxed 500 set

- Added `data/VDR-data/train/M3VQA/5_reliability_filter.py` to audit Step 4 samples with `gemini-3.1-flash-lite-preview`, checkpointing every sample and rejecting risky QA pairs.
- Step 5 completed for 1500 samples with 0 judge failures; strict keep set has 36 samples.
- Added `6_select_relaxed_reliable.py` to build a practical 500-sample set by excluding high-risk tags (`answer_mismatch`, `counting`, binary/judgment, malformed, vague, multi_possible) while allowing conservative comparative/ambiguous/search-friendliness warnings.
- Generated `m3vqa_filtered_relaxed500.jsonl` and `m3vqa_train_relaxed500.jsonl` with 500 FVQA-style samples.
- Verified output fields are `answer/id/images/question` and no empty answers.
## 2026-05-08 - M3VQA relaxed500 multimodal question rewrite

- Added `data/VDR-data/train/M3VQA/7_rewrite_questions.py` to rewrite relaxed500 questions with Gemini while sending the source image as multimodal input.
- Prompt now requires preserving visual referents and keeping the original question if rewriting would change the answer or referent.
- Generated `m3vqa_train_relaxed500_rewritten.jsonl` and `m3vqa_filtered_relaxed500_rewritten.jsonl`; 448/500 questions changed.
- Verified output remains 500 FVQA-style rows, no empty answers, and no answer leakage by exact lowercase match.
- Candidate-list wording decreased substantially (`in picture, A and B`: 84 -> 4; `which of`: 51 -> 35).

### 2026-05-20 - MC-Search API model switch to Qwen3-VL-235B

- Updated SFT pipeline defaults so Step1 direct-answer agent and judge both use `Qwen3-VL-235B-A22B-Instruct` at `http://200.20.7.6:8003/v1`.
- Updated MC-Search Step2 API rollout configs and convenience launchers to use `Qwen3-VL-235B-A22B-Instruct` for the agentic model.
- Verified `common/config.py` and Step1 compile, and Step2 launcher shell syntax passes `bash -n`.
