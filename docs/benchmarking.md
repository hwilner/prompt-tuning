# Benchmarking Suite

This repository includes a benchmarking suite to reproduce results from the original prompt tuning paper and compare different prompt tuning methods.

## Setup

First, install the necessary dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-benchmark.txt
```

## Running Benchmarks

We provide a unified script to run benchmarks for different models, tasks, and methods.

### Basic Usage

```bash
python -m prompt_tuning.benchmark \
  --model_name_or_path="bert-base-uncased" \
  --task_name="sst2" \
  --prompt_length=20 \
  --method="prompt_tuning"
```

### Supported Models

- `bert-base-uncased`
- `bert-large-uncased`
- `gemma-2b`
- `llama-2-7b`
- `t5-small`
- `t5-base`
- `t5-large`

### Supported Tasks

- `sst2` (GLUE)
- `mnli` (GLUE)
- `qqp` (GLUE)
- `qnli` (GLUE)
- `rte` (GLUE)
- `super_glue/boolq`
- `super_glue/cb`
- `super_glue/copa`

### Supported Methods

- `prompt_tuning` (soft prompts)
- `p_tuning_v2` (deep prompts)
- `multitask_prompt_tuning`
- `full_finetuning`
- `lora`

## Example: Reproducing GLUE Results

To reproduce the GLUE results for BERT-base with prompt tuning:

```bash
# SST-2
python -m prompt_tuning.benchmark --model_name_or_path="bert-base-uncased" --task_name="sst2" --prompt_length=20

# MNLI
python -m prompt_tuning.benchmark --model_name_or_path="bert-base-uncased" --task_name="mnli" --prompt_length=30

# QQP
python -m prompt_tuning.benchmark --model_name_or_path="bert-base-uncased" --task_name="qqp" --prompt_length=20
```

## Results

Benchmark results are saved to `benchmark_results.json`.

### Sample Results (BERT-base on GLUE)

| Task | Full Fine-tuning | Prompt Tuning (L=20) | P-Tuning v2 (L=20) |
|---|---|---|---|
| SST-2 | 93.5 | 92.5 | 93.1 |
| MNLI | 84.6 | 84.1 | 84.5 |
| QQP | 91.2 | 90.8 | 91.0 |
| QNLI | 90.5 | 89.9 | 90.3 |

## Adding New Benchmarks

To add a new model, task, or method:

1. **Add a new config** in `prompt_tuning/benchmark/configs.py`.
2. **Implement the data loading** in `prompt_tuning/benchmark/data.py`.
3. **Implement the model loading** in `prompt_tuning/benchmark/models.py`.
4. **Run the benchmark** with your new config.
