<div align="center">

# SimpleSearch-VL: A Simple Agentic System for Multimodal Deep Search Agents


**Evidence-Verified SFT · Factorized Adaptive Rollout · Self-Summarized Visit**

[![Paper](https://img.shields.io/badge/Paper-arXiv-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.31504)

[![SFT-5K dataset on ModelScope](https://img.shields.io/badge/Dataset-SFT--5K-2563EB?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48ZWxsaXBzZSBjeD0iMTIiIGN5PSI1IiByeD0iOSIgcnk9IjMiLz48cGF0aCBkPSJNMyA1djE0YzAgNCAxOCA0IDE4IDBWNU0zIDEyYzAgNCAxOCA0IDE4IDAiLz48L3N2Zz4%3D&logoColor=white)](https://www.modelscope.cn/datasets/dmmm997/SimpleSearch-VL-SFT-5K)
[![RL-2K dataset on ModelScope](https://img.shields.io/badge/Dataset-RL--2K-2563EB?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48ZWxsaXBzZSBjeD0iMTIiIGN5PSI1IiByeD0iOSIgcnk9IjMiLz48cGF0aCBkPSJNMyA1djE0YzAgNCAxOCA0IDE4IDBWNU0zIDEyYzAgNCAxOCA0IDE4IDAiLz48L3N2Zz4%3D&logoColor=white)](https://www.modelscope.cn/datasets/dmmm997/SimpleSearch-VL-RL-2K)
[![Evaluation dataset on ModelScope](https://img.shields.io/badge/Dataset-Evaluation-2563EB?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48ZWxsaXBzZSBjeD0iMTIiIGN5PSI1IiByeD0iOSIgcnk9IjMiLz48cGF0aCBkPSJNMyA1djE0YzAgNCAxOCA0IDE4IDBWNU0zIDEyYzAgNCAxOCA0IDE4IDAiLz48L3N2Zz4%3D&logoColor=white)](https://www.modelscope.cn/datasets/dmmm997/SimpleSearch-VL-EvalData)

[![8B weights on ModelScope](https://img.shields.io/badge/Weights-8B-059669?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJtMTIgMiAxMCA1djEwbC0xMCA1LTEwLTVWN1ptMCAxMEwyIDdtMTAgNUwyMiA3TTEyIDEydjEwIi8%2BPC9zdmc%2B&logoColor=white)](https://www.modelscope.cn/models/dmmm997/SimpleSearch-VL-8B)
[![30B-A3B weights on ModelScope](https://img.shields.io/badge/Weights-30B--A3B-059669?style=for-the-badge&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj48cGF0aCBkPSJtMTIgMiAxMCA1djEwbC0xMCA1LTEwLTVWN1ptMCAxMEwyIDdtMTAgNUwyMiA3TTEyIDEydjEwIi8%2BPC9zdmc%2B&logoColor=white)](https://www.modelscope.cn/models/dmmm997/SimpleSearch-VL-30B-A3B)


</div>

SimpleSearch-VL is an agentic system for **multimodal deep search**, built on Qwen3-VL. It combines evidence-verified reasoning, region-level image search, and **Factorized Adaptive Rollout (FAR)**, using only 5K SFT trajectories and 2K RL prompts.

![SimpleSearch-VL performance](figs/performance.png)

## ✨ Highlights

- **Factorized Adaptive Rollout (FAR)**: allocates prompt expansion and rollouts according to reward signals.
- **Evidence-verified reasoning**: checks textual evidence and matched image thumbnails before answering.
- **Self-summarized visit**: uses the policy model for webpage summarization, without an extra model.
- **Simple tool stack**: Serper, JINA Reader, and OSS.
- **Efficient training**: 8B and 30B-A3B variants trained in roughly 1–2 days on 8×H200, with average gains of **+15.8 / +16.0** over their Qwen3-VL agentic baselines.
- **Strong results**: the 30B-A3B variant outperforms agentic Gemini-3-Pro on MMSearch, MMSearch+, BrowseComp-VL, FVQA, and LiveVQA.

![SimpleSearch-VL workflow](figs/markov_vis.jpg)

## 📁 Repository Layout

```text
SimpleSearch-VL/
|-- data/SimpleSearch-VL/         # Datasets
|-- checkpoints/                 # Local model weights
|-- figs/                        # Figures
|-- rllm/
|   |-- simple_agentic_search/   # Agent, tools, configs, and RL/eval launchers
|   |-- rllm/                    # Agent runtime
|   |-- verl/                    # RL trainer
|-- ms-swift/                    # SFT framework and launchers
|-- Megatron-LM/
|-- mbridge/
```

## 🛠️ Installation

Use a CUDA-compatible Python environment with Bash 4.4+ and `rsync`.

```bash
export PROJECT_ROOT=/path/to/SimpleSearch-VL
cd "$PROJECT_ROOT"
bash env_start.sh

# For SFT:
python -m pip install -e ./ms-swift
```

[Reference package versions](pip_list_simplesearch_vl.txt).

## ⚙️ Configuration

Create `rllm/.env` and fill in your service credentials using the [environment template](rllm/.env.example):

```bash
cd "$PROJECT_ROOT/rllm"
[ -e .env ] || (umask 077; cp .env.example .env)
```

See the [configuration guide](rllm/simple_agentic_search/config/README.md) for templates and parameters. Copy a template into `rllm/simple_agentic_search/config/local/`, customize it, and pass your YAML to the launcher. Paths in YAML are relative to `rllm/`; keep credentials in `.env`.

## 📂 Datasets

Download each dataset into the directory below and ensure its JSONL image paths resolve locally.

| Dataset | Download | Local directory |
| --- | --- | --- |
| SFT-5K | [ModelScope](https://www.modelscope.cn/datasets/dmmm997/SimpleSearch-VL-SFT-5K) | `$PROJECT_ROOT/data/SimpleSearch-VL/SimpleSearch-VL-SFT-5K/` |
| RL-2K | [ModelScope](https://www.modelscope.cn/datasets/dmmm997/SimpleSearch-VL-RL-2K) | `$PROJECT_ROOT/data/SimpleSearch-VL/SimpleSearch-VL-RL-2K/` |
| Evaluation | [ModelScope](https://www.modelscope.cn/datasets/dmmm997/SimpleSearch-VL-EvalData) | `$PROJECT_ROOT/data/SimpleSearch-VL/Evaluation/` |

Expected JSONL files under `$PROJECT_ROOT/data/SimpleSearch-VL/`:

```text
SimpleSearch-VL-SFT-5K/sft_ms_swift.jsonl
SimpleSearch-VL-RL-2K/train.jsonl
Evaluation/browsecomp_vl/browsecomp_vl.jsonl
Evaluation/fvqa_test/fvqa_test.jsonl
Evaluation/livevqa_research_preview/livevqa_research_preview.jsonl
Evaluation/mmsearch_end2end_only_image/mmsearch_end2end_only_image.jsonl
Evaluation/mmsearch_plus/mmsearch_plus.jsonl
Evaluation/simplevqa_en/simplevqa_en.jsonl
```

## 🏋️ Checkpoints

Released RL checkpoints:

| Model | Download | Local directory |
| --- | --- | --- |
| 8B | [ModelScope](https://www.modelscope.cn/models/dmmm997/SimpleSearch-VL-8B) | `$PROJECT_ROOT/checkpoints/SimpleSearch-VL-8B-RL/` |
| 30B-A3B | [ModelScope](https://www.modelscope.cn/models/dmmm997/SimpleSearch-VL-30B-A3B) | `$PROJECT_ROOT/checkpoints/SimpleSearch-VL-30B-A3B-RL/` |

Use these directories or update `model_path` in your YAML. Intermediate SFT checkpoints are not provided.

## 🏋️ Supervised Fine-Tuning

Place the original [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) or [Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) checkpoint under `$PROJECT_ROOT/checkpoints/`.

```bash
cd "$PROJECT_ROOT"

# 8B
bash ms-swift/run/SFT_Qwen3VL_8B_megatron_lr2e5_3ep.sh

# 30B-A3B
bash ms-swift/run/SFT_Qwen3VL_30B_A3B_megatron_lr1e5_3ep.sh
```

## 🚀 Reinforcement Learning

Copy the templates and set `model_path` to an original Qwen3-VL-Instruct checkpoint or your SFT output:

```bash
cd "$PROJECT_ROOT/rllm"
mkdir -p simple_agentic_search/config/local
cp -n simple_agentic_search/config/qwen3vl8b_sft_dapo_2x_dynroll_4_6_150step.yaml simple_agentic_search/config/local/train_8b.yaml
cp -n simple_agentic_search/config/qwen3vl30b_sft_dapo_2x_dynroll_4_6_150step.yaml simple_agentic_search/config/local/train_30b.yaml
```

### Single node

```bash
cd "$PROJECT_ROOT/rllm"

# 8B
NNODES=1 bash simple_agentic_search/run/run_multi_node.sh simple_agentic_search/config/local/train_8b.yaml

# 30B-A3B
NNODES=1 bash simple_agentic_search/run/run_multi_node.sh simple_agentic_search/config/local/train_30b.yaml
```

### Multiple nodes

```bash
cd "$PROJECT_ROOT/rllm"

# Use dedicated nodes: the launcher restarts local Ray processes.
# Node 0
NODE_RANK=0 NNODES=2 MASTER_ADDR="<head-node-ip>" bash simple_agentic_search/run/run_multi_node.sh simple_agentic_search/config/local/train_30b.yaml

# Node 1
NODE_RANK=1 NNODES=2 MASTER_ADDR="<head-node-ip>" bash simple_agentic_search/run/run_multi_node.sh simple_agentic_search/config/local/train_30b.yaml
```

## 📊 Evaluation

### Local checkpoints

Use the released weights and both the RL and evaluation datasets listed above.

```bash
cd "$PROJECT_ROOT/rllm"

# 8B
NNODES=1 bash simple_agentic_search/run/run_multi_node.sh simple_agentic_search/config/eval_8B.yaml

# 30B-A3B
NNODES=1 bash simple_agentic_search/run/run_multi_node.sh simple_agentic_search/config/eval_30B.yaml
```

### API models

Copy the API template for a vision-capable, OpenAI-compatible endpoint:

```bash
cd "$PROJECT_ROOT/rllm"
mkdir -p simple_agentic_search/config/local
cp -n simple_agentic_search/config/eval_apimodel.yaml simple_agentic_search/config/local/eval_api.yaml
```

Fill `agent_model_api.base_url` and `model` in the copied YAML, and `AGENT_API_KEY` in `.env`, then run:

```bash
bash simple_agentic_search/run/run.sh simple_agentic_search/config/local/eval_api.yaml
```

## 🔧 Tools

- **`text_search`**: web search.
- **`image_search`**: region-level reverse image search with thumbnails for verification.
- **`visit`**: webpage reading and policy-model summarization.
- **`zoom_in`** (optional): visual region inspection; enable it in your YAML.

## 🔎 Visualization

An example of multi-image search and evidence verification:

![SimpleSearch-VL example](figs/vis_sample.jpg)

## 🙌 Acknowledgements

We thank [Vision-DeepResearch](https://github.com/Osilly/Vision-DeepResearch) for its codebase, [OpenSearch-VL](https://github.com/shawn0728/OpenSearch-VL) for part of the data, and [rLLM](https://github.com/rllm-org/rllm) and [ms-swift](https://github.com/modelscope/ms-swift) for the training infrastructure.

## 📄 Citation

If you use SimpleSearch-VL, please cite our [paper](https://arxiv.org/abs/2606.31504):

```bibtex
@misc{dai2026simplesearchvlsimplerecipemultimodal,
      title={SimpleSearch-VL: A Simple Recipe for Multimodal Agentic Deep Search},
      author={Ming Dai and Zhihong Lu and Jinjie Gu and Jiedong Zhuang and Yefeng Liu and Wankou Yang and Jian Wang and Chunhua Shen},
      year={2026},
      eprint={2606.31504},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.31504},
}
```
