# Welcome to the Enhanced Prompt Tuning Library

This repository is a community-driven, enhanced version of Google Research's original [prompt-tuning](https://github.com/google-research/prompt-tuning) library. It extends the original work with support for new models, advanced prompt tuning techniques, and comprehensive documentation.

## Key Enhancements

### 1. New Model Support

This fork adds prompt tuning support for several new models beyond the original T5:

- **Gemma**: Google's latest open-weight models (270M to 27B)
- **BERT**: Encoder-only models for classification (base, large, DistilBERT)
- **LLaMA-2**: Meta's popular open-source models (7B, 13B, 70B)

### 2. Advanced Prompt Tuning Techniques

We've implemented several cutting-edge techniques from recent research:

- **P-Tuning v2 (Deep Prompt Tuning)**: Apply prompts at every layer
- **Multi-Task Prompt Tuning**: Learn shared and task-specific prompts
- **Dynamic Prompts**: Generate prompts conditioned on input
- **Visual Prompt Tuning**: Extend prompt tuning to vision-language models

### 3. Comprehensive Documentation

- **Tutorials**: Step-by-step guides for all models and techniques
- **Model Zoo**: Pretrained prompts for common tasks
- **Benchmarking**: Scripts to reproduce results and compare methods
- **API Reference**: Complete documentation for all modules

## Getting Started

### Installation

```bash
# Clone this repository
git clone https://github.com/hwilner/prompt-tuning.git
cd prompt-tuning

# Install dependencies
pip install -r requirements.txt
```

### Quick Start: Gemma Prompt Tuning

```python
from prompt_tuning.gemma import prompts, train
from gemma import gm

# 1. Load base model
model = gm.nn.Gemma3_2B()
base_params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_2B_IT)

# 2. Create prompt-tuned model
prompt_config = prompts.PromptConfig(prompt_length=20, embed_dim=2048)
prompt_model = prompts.create_prompt_model(model, prompt_config)

# 3. Train
prompt_params, history = train.train_prompt(
    model=prompt_model,
    prompt_params=prompt_model.init_prompt_params(jax.random.PRNGKey(0)),
    base_params=base_params,
    train_ds=your_dataset,
)
```

## Tutorials

- [Tutorial 1: Prompt Tuning for Text Classification with BERT](./tutorials/bert_classification.md)
- [Tutorial 2: Prompt Tuning for Text Generation with Gemma](./tutorials/gemma_generation.md)
- [Tutorial 3: Deep Prompt Tuning (P-Tuning v2)](./tutorials/p_tuning_v2.md)
- [Tutorial 4: Multi-Task Prompt Tuning](./tutorials/multitask_prompts.md)

## Model Zoo

We provide a collection of pretrained soft prompts for common tasks and datasets. These can be used as a starting point for your own fine-tuning or for direct inference.

[Browse the Model Zoo](./model_zoo.md)

## Benchmarking

This repository includes a benchmarking suite to reproduce results from the original paper and compare different prompt tuning methods.

[Run Benchmarks](./benchmarking.md)

## API Reference

- [Gemma Module](./api/gemma.md)
- [BERT Module](./api/bert.md)
- [LLaMA-2 Module](./api/llama2.md)
- [Experimental Techniques](./api/experimental.md)

## Contributing

We welcome contributions! Please see our [contributing guide](./CONTRIBUTING.md) for details.

## License

Apache License 2.0
