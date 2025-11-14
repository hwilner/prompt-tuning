# Prompt Tuning for LLaMA-2

This module provides prompt tuning capabilities for Meta's LLaMA-2 models. LLaMA-2 is a family of open-source decoder-only transformer models ranging from 7B to 70B parameters.

## Overview

**Prompt tuning** is a parameter-efficient fine-tuning method that learns soft prompts (continuous embeddings) while keeping the base model frozen. For LLaMA-2, this means tuning only 0.001-0.01% of parameters while achieving strong performance.

## Features

- ✅ **Soft prompt tuning** for all LLaMA-2 model sizes (7B, 13B, 70B)
- ✅ **Compatible with RoPE** position encoding
- ✅ **Supports GQA** (Grouped Query Attention)
- ✅ **Multiple initialization** strategies (random, from vocab, from text)
- ✅ **Adapters** for popular JAX LLaMA-2 implementations
- ✅ **Easy integration** with existing workflows

## Installation

This module is designed to work with existing JAX/Flax LLaMA-2 implementations. You'll need one of:

### Option 1: ayaka14732/llama-2-jax (Recommended)

```bash
# Clone the repository
git clone https://github.com/ayaka14732/llama-2-jax.git
cd llama-2-jax

# Follow their installation instructions
# Install JAX, PyTorch, and dependencies
pip install -r requirements.txt
```

### Option 2: Hugging Face Transformers (Flax)

```bash
pip install transformers[flax]
pip install jax jaxlib  # or jax[cuda] for GPU
```

### Option 3: young-geng/EasyLM

```bash
git clone https://github.com/young-geng/EasyLM.git
# Follow their installation instructions
```

## Quick Start

### With ayaka14732/llama-2-jax

```python
import jax
from llama_2_jax import LlamaModel, load_params
from prompt_tuning.llama2 import prompts

# 1. Load base LLaMA-2 model
model = LlamaModel(...)
base_params = load_params('path/to/llama2-7b')

# 2. Create prompt-tuned model
prompt_config = prompts.PromptConfig(
    prompt_length=20,
    embed_dim=4096,  # LLaMA-2 7B
)
prompt_model = prompts.LLaMA2JaxAdapter.create_prompt_model(
    model, prompt_config
)

# 3. Initialize prompt parameters
rng = jax.random.PRNGKey(42)
prompt_params = prompt_model.init_prompt_params(rng)

# 4. Train (see examples for full training loop)
# ... training code ...

# 5. Use for inference
output = prompt_model.apply(
    {'params': {**base_params, **prompt_params}},
    input_ids,
)
```

### With Hugging Face Transformers

```python
import jax
from transformers import FlaxLlamaForCausalLM
from prompt_tuning.llama2 import prompts

# 1. Load base model
model = FlaxLlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')

# 2. Create prompt-tuned model
prompt_config = prompts.PromptConfig(
    prompt_length=20,
    embed_dim=4096,
)
prompt_model = prompts.HuggingFaceFlaxAdapter.create_prompt_model(
    model, prompt_config
)

# 3. Train and use
# ... similar to above ...
```

## Model Configurations

| Model | Parameters | d_model | Layers | Heads (KV) | GQA Ratio | d_ff |
|-------|------------|---------|--------|------------|-----------|------|
| LLaMA-2 7B | 6.7B | 4096 | 32 | 32 (32) | 1:1 | 11008 |
| LLaMA-2 13B | 13B | 5120 | 40 | 40 (40) | 1:1 | 13824 |
| LLaMA-2 70B | 69B | 8192 | 80 | 64 (8) | 8:1 | 28672 |

**Configuration for prompt tuning:**

```python
# LLaMA-2 7B
prompt_config = prompts.PromptConfig(prompt_length=20, embed_dim=4096)

# LLaMA-2 13B
prompt_config = prompts.PromptConfig(prompt_length=20, embed_dim=5120)

# LLaMA-2 70B
prompt_config = prompts.PromptConfig(prompt_length=20, embed_dim=8192)
```

## Architecture Details

### Key Differences from Other Models

LLaMA-2 uses several architectural innovations:

1. **Grouped Query Attention (GQA)**: Reduces KV cache size
   - 7B/13B: 1:1 ratio (standard MHA)
   - 70B: 8:1 ratio (8 query heads per KV head)

2. **RoPE (Rotary Position Embeddings)**: Relative position encoding
   - Applied to query and key vectors
   - Enables better length extrapolation

3. **RMS Normalization**: More efficient than LayerNorm
   - No bias term
   - Only rescaling, no recentering

4. **SwiGLU Activation**: In feed-forward layers
   - Gated linear unit with Swish activation
   - Better than ReLU/GELU

### Prompt Integration

For LLaMA-2's decoder-only architecture, prompts are prepended to input:

```
[prompt_1, prompt_2, ..., prompt_n, token_1, token_2, ..., token_m]
```

**Position Encoding (RoPE)**:
```
Prompt positions: [0, 1, 2, ..., prompt_length-1]
Input positions:  [prompt_length, prompt_length+1, ..., prompt_length+seq_len-1]
```

**Attention Masking**:
```
- Prompts can attend to each other (bidirectional)
- Input tokens can attend to all prompts
- Input tokens use causal masking among themselves
```

## Training

### Basic Training Loop

```python
from prompt_tuning.llama2 import prompts
import optax

# Create optimizer for prompts only
optimizer = optax.adam(learning_rate=0.3)
opt_state = optimizer.init(prompt_params)

# Training step
def train_step(prompt_params, base_params, batch):
    def loss_fn(prompt_params):
        full_params = {**base_params, **prompt_params}
        logits = prompt_model.apply(
            {'params': full_params},
            batch['input_ids'],
        )
        # Compute loss
        loss = compute_loss(logits, batch['labels'])
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(prompt_params)
    updates, opt_state = optimizer.update(grads, opt_state)
    prompt_params = optax.apply_updates(prompt_params, updates)
    
    return prompt_params, loss

# Training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        prompt_params, loss = train_step(prompt_params, base_params, batch)
```

### Hyperparameters

| Hyperparameter | Recommended Value | Notes |
|----------------|-------------------|-------|
| Prompt Length | 10-50 | Longer for complex tasks |
| Learning Rate | 0.1-0.5 | Higher than full fine-tuning |
| Batch Size | 4-16 | Depends on GPU memory |
| Epochs | 3-10 | Task-dependent |
| Weight Decay | 0.0-0.01 | Optional regularization |

## Initialization Strategies

### Random Initialization (Default)

```python
prompt_config = prompts.PromptConfig(
    prompt_length=20,
    embed_dim=4096,
    init_scale=0.5,  # Standard deviation
)
```

### From Vocabulary

```python
# Get vocabulary embeddings from model
vocab_embeddings = get_vocab_embeddings(model, params)

# Initialize from random vocabulary tokens
prompt_embeds = prompts.init_from_vocab(
    vocab_embeddings,
    prompt_length=20,
    rng=jax.random.PRNGKey(0),
)
```

### From Text

```python
# Initialize from task description
prompt_embeds = prompts.init_from_text(
    text="Summarize the following text:",
    tokenizer=tokenizer,
    vocab_embeddings=vocab_embeddings,
    prompt_length=20,
    rng=jax.random.PRNGKey(0),
)
```

## Performance

### Parameter Efficiency

For LLaMA-2 7B (6.7B parameters):

| Method | Trainable Params | Percentage |
|--------|------------------|------------|
| Full Fine-tuning | 6.7B | 100% |
| LoRA (r=16) | ~16M | 0.24% |
| **Prompt Tuning (L=20)** | **~80K** | **0.0012%** |
| **Prompt Tuning (L=100)** | **~400K** | **0.006%** |

### Expected Results

On standard benchmarks, prompt tuning typically achieves:

- **85-95%** of full fine-tuning performance
- **Better than** few-shot prompting
- **Comparable to** LoRA on many tasks

## Integration with Existing Implementations

### ayaka14732/llama-2-jax

```python
from llama_2_jax import LlamaModel, load_params
from prompt_tuning.llama2 import LLaMA2JaxAdapter

# Load model
model = LlamaModel(...)
params = load_params(...)

# Create prompt model
prompt_model = LLaMA2JaxAdapter.create_prompt_model(
    model,
    prompt_config,
)

# Get embeddings for initialization
vocab_embeds = LLaMA2JaxAdapter.get_embeddings(model, params)
```

### Hugging Face Transformers

```python
from transformers import FlaxLlamaForCausalLM
from prompt_tuning.llama2 import HuggingFaceFlaxAdapter

# Load model
model = FlaxLlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')

# Create prompt model
prompt_model = HuggingFaceFlaxAdapter.create_prompt_model(
    model,
    prompt_config,
)

# Get embeddings
vocab_embeds = HuggingFaceFlaxAdapter.get_embeddings(model, model.params)
```

## Comparison with Gemma

Both LLaMA-2 and Gemma are decoder-only models, but with differences:

| Aspect | LLaMA-2 | Gemma |
|--------|---------|-------|
| Attention | GQA (70B) | MQA/GQA |
| Normalization | RMS Norm | RMS Norm |
| Position Encoding | RoPE | RoPE |
| Activation | SwiGLU | GeGLU |
| Vocabulary | 32K | 256K |
| License | Custom (Llama 2) | Gemma Terms |

**Prompt tuning works identically** for both since they share the same decoder-only architecture!

## Troubleshooting

### Out of Memory

- Reduce batch size
- Use gradient accumulation
- Try a smaller model (7B instead of 70B)
- Reduce prompt length

### Poor Performance

- Increase prompt length
- Increase learning rate
- Train for more epochs
- Try text initialization
- Use a larger model

### Slow Training

- Use TPU instead of GPU
- Increase batch size
- Enable JAX JIT compilation
- Use model parallelism for 70B

## Examples

See `examples/` directory for:
- Text classification
- Text generation
- Instruction following
- Multi-task learning

## Resources

- [LLaMA-2 Paper](https://arxiv.org/abs/2307.09288)
- [ayaka14732/llama-2-jax](https://github.com/ayaka14732/llama-2-jax)
- [Hugging Face LLaMA-2](https://huggingface.co/meta-llama)
- [Original Prompt Tuning Paper](https://arxiv.org/abs/2104.08691)

## Citation

```bibtex
@article{touvron2023llama2,
  title={Llama 2: Open Foundation and Fine-Tuned Chat Models},
  author={Touvron, Hugo and Martin, Louis and Stone, Kevin and others},
  journal={arXiv preprint arXiv:2307.09288},
  year={2023}
}

@article{lester2021power,
  title={The Power of Scale for Parameter-Efficient Prompt Tuning},
  author={Lester, Brian and Al-Rfou, Rami and Constant, Noah},
  journal={EMNLP},
  year={2021}
}
```

## License

Apache License 2.0 - See LICENSE file for details.
