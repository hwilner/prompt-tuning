# Prompt Tuning for Gemma

This module provides prompt tuning capabilities for Google DeepMind's Gemma models. Gemma is a family of open-weight decoder-only transformer models based on Gemini research and technology.

## Overview

**Prompt tuning** is a parameter-efficient fine-tuning method that learns soft prompts (continuous embeddings) while keeping the base model frozen. This approach:

- Requires tuning only 0.01-0.1% of model parameters
- Achieves comparable performance to full fine-tuning
- Enables easy multi-task learning (one prompt per task)
- Reduces storage and deployment costs

## Features

- ✅ **Soft prompt tuning** for all Gemma model sizes (270M, 2B, 4B, 7B, 27B)
- ✅ **Compatible with RoPE** position encoding
- ✅ **Causal attention** masking for decoder-only architecture
- ✅ **Multiple initialization** strategies (random, from vocab, from text)
- ✅ **Training utilities** with JAX/Flax
- ✅ **Easy integration** with existing Gemma workflows

## Installation

```bash
# Install Gemma library
pip install gemma

# Install JAX (CPU, GPU, or TPU)
# See https://jax.readthedocs.io/en/latest/installation.html

# Install additional dependencies
pip install optax flax tqdm
```

## Quick Start

### Basic Usage

```python
import jax
from gemma import gm
from prompt_tuning.gemma import prompts, train

# 1. Load pretrained Gemma model
model = gm.nn.Gemma3_2B()
base_params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_2B_IT)

# 2. Create prompt-tuned model
prompt_config = prompts.PromptConfig(
    prompt_length=20,      # Number of prompt tokens
    embed_dim=2048,        # Gemma 2B embedding dimension
    init_scale=0.5,        # Initialization scale
)
prompt_model = prompts.create_prompt_model(model, prompt_config)

# 3. Initialize prompt parameters
rng = jax.random.PRNGKey(42)
prompt_params = prompt_model.init_prompt_params(rng)

# 4. Train (see examples for full training loop)
trained_params, history = train.train_prompt(
    model=prompt_model,
    prompt_params=prompt_params,
    base_params=base_params,
    train_ds=your_train_dataset,
    num_epochs=3,
    learning_rate=0.3,
)

# 5. Use for inference
output = prompt_model.apply(
    {'params': {**base_params, **trained_params}},
    input_ids,
)
```

### Model Sizes and Configurations

| Model | Embed Dim | Layers | Heads | Params | Checkpoint Path |
|-------|-----------|--------|-------|--------|-----------------|
| Gemma3-270M | 1408 | 18 | 11 | 270M | `GEMMA3_270M_IT` |
| Gemma3-2B | 2048 | 26 | 16 | 2B | `GEMMA3_2B_IT` |
| Gemma3-4B | 2816 | 32 | 22 | 4B | `GEMMA3_4B_IT` |
| Gemma2-7B | 3072 | 28 | 16 | 7B | `GEMMA2_7B_IT` |
| Gemma2-27B | 4608 | 46 | 32 | 27B | `GEMMA2_27B_IT` |

## Examples

### Text Classification

```python
# See examples/simple_classification.py for a complete example
python examples/simple_classification.py \
  --model_size=2b \
  --prompt_length=20 \
  --num_epochs=3 \
  --learning_rate=0.3
```

### Custom Initialization

```python
# Initialize from vocabulary embeddings
vocab_embeddings = model.embedder.embed.embedding
prompt_embeds = prompts.init_from_vocab(
    vocab_embeddings,
    prompt_length=20,
    rng=jax.random.PRNGKey(0),
)

# Initialize from text
prompt_embeds = prompts.init_from_text(
    text="Classify this text:",
    tokenizer=tokenizer,
    vocab_embeddings=vocab_embeddings,
    prompt_length=20,
    rng=jax.random.PRNGKey(0),
)
```

### Multi-Task Learning

```python
# Train separate prompts for different tasks
tasks = ['classification', 'generation', 'qa']
prompt_params = {}

for task in tasks:
    # Train prompt for each task
    params, _ = train.train_prompt(
        model=prompt_model,
        prompt_params=initial_params,
        base_params=base_params,
        train_ds=task_datasets[task],
        num_epochs=3,
    )
    prompt_params[task] = params

# Use task-specific prompts at inference
output = prompt_model.apply(
    {'params': {**base_params, **prompt_params['classification']}},
    input_ids,
)
```

## Architecture Details

### Prompt Insertion

For Gemma's decoder-only architecture, prompts are prepended to input embeddings:

```
[prompt_1, prompt_2, ..., prompt_n, token_1, token_2, ..., token_m]
```

### Position Encoding

Gemma uses Rotary Position Embeddings (RoPE). Position indices are adjusted:

```
Prompt positions: [0, 1, 2, ..., prompt_length-1]
Input positions:  [prompt_length, prompt_length+1, ..., prompt_length+seq_len-1]
```

### Attention Masking

Causal attention is applied with prompts:

```
- Prompts can attend to each other (bidirectional)
- Input tokens can attend to all prompts
- Input tokens use causal masking among themselves
```

## Hyperparameters

### Recommended Settings

| Hyperparameter | Recommended Value | Notes |
|----------------|-------------------|-------|
| Prompt Length | 10-100 | Longer prompts = more capacity but slower |
| Learning Rate | 0.1-0.5 | Higher than typical fine-tuning |
| Batch Size | 8-32 | Depends on available memory |
| Epochs | 3-10 | Task-dependent |
| Weight Decay | 0.0-0.01 | Optional regularization |

### Prompt Length Guidelines

- **Simple tasks** (sentiment analysis): 10-20 tokens
- **Medium tasks** (NLI, QA): 20-50 tokens
- **Complex tasks** (summarization): 50-100 tokens

## Performance

### Parameter Efficiency

For Gemma 2B (2.5B parameters):

| Method | Trainable Params | Percentage |
|--------|------------------|------------|
| Full Fine-tuning | 2.5B | 100% |
| LoRA (r=16) | ~4M | 0.16% |
| **Prompt Tuning (L=20)** | **~40K** | **0.0016%** |

### Expected Results

On standard benchmarks, prompt tuning typically achieves:

- **90-95%** of full fine-tuning performance
- **Better than** few-shot prompting
- **Comparable to** LoRA on many tasks

## Advanced Features

### Deep Prompt Tuning (P-Tuning v2)

For even better performance, use deep prompts at multiple layers:

```python
from prompt_tuning.experimental import p_tuning_v2

# Create deep prompt model
deep_prompt_model = p_tuning_v2.DeepPromptModel(
    base_model=model,
    prompt_length=20,
    num_layers=26,  # Gemma 2B has 26 layers
)
```

### Dynamic Prompts

Use input-conditioned prompts:

```python
from prompt_tuning.experimental import dynamic_prompts

# Create dynamic prompt model
dynamic_model = dynamic_prompts.DynamicPromptModel(
    base_model=model,
    prompt_length=20,
    num_prototypes=10,
)
```

## Troubleshooting

### Out of Memory

- Reduce batch size
- Use gradient accumulation
- Use a smaller model (270M or 2B)
- Reduce prompt length

### Poor Performance

- Increase prompt length
- Increase learning rate
- Train for more epochs
- Try different initialization strategies
- Use deep prompt tuning

### Slow Training

- Use TPU instead of GPU/CPU
- Increase batch size
- Use gradient accumulation
- Enable JAX JIT compilation

## Comparison with T5 Prompt Tuning

| Aspect | T5 (Encoder-Decoder) | Gemma (Decoder-Only) |
|--------|----------------------|----------------------|
| Architecture | Encoder-Decoder | Decoder-Only |
| Prompt Location | Encoder input | Decoder input |
| Position Encoding | Relative | RoPE |
| Attention | Bidirectional (encoder) | Causal |
| Use Cases | Translation, summarization | Generation, chat |

## Citation

If you use this implementation, please cite:

```bibtex
@article{lester2021power,
  title={The Power of Scale for Parameter-Efficient Prompt Tuning},
  author={Lester, Brian and Al-Rfou, Rami and Constant, Noah},
  journal={EMNLP},
  year={2021}
}

@article{gemma2024,
  title={Gemma: Open Models Based on Gemini Research and Technology},
  author={Gemma Team},
  journal={arXiv preprint arXiv:2403.08295},
  year={2024}
}
```

## Resources

- [Gemma Documentation](https://gemma-llm.readthedocs.io)
- [Gemma GitHub](https://github.com/google-deepmind/gemma)
- [Original Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)
- [JAX Documentation](https://jax.readthedocs.io)

## License

Apache License 2.0 - See LICENSE file for details.
