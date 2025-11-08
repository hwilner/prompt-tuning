# Hybrid Prompt-Adapter Tuning

## Overview

Hybrid Prompt-Adapter Tuning is a novel parameter-efficient fine-tuning approach that combines the strengths of two popular methods: prompt tuning and adapter modules. This technique aims to achieve better performance than either method alone by providing a more flexible adaptation mechanism that operates at multiple levels of the transformer architecture.

## Motivation

While prompt tuning has proven effective for adapting large language models to new tasks with minimal parameter updates, it primarily modifies the input representation. Adapter modules, on the other hand, insert small bottleneck layers within the transformer blocks, allowing for more fine-grained control over the model's intermediate representations. By combining these approaches, we hypothesize that we can achieve:

1. **Better task adaptation**: Prompts guide the model's attention at the input level, while adapters refine the internal representations.
2. **Improved parameter efficiency**: The combined approach may require fewer total parameters than full fine-tuning while maintaining competitive performance.
3. **Greater flexibility**: Different combinations of prompt length and adapter size can be tuned for specific tasks and model sizes.

## Technical Approach

### Architecture

The hybrid approach integrates prompts and adapters as follows:

1. **Input Layer Adaptation**: Trainable prompt embeddings are prepended to the input sequence, similar to standard prompt tuning.
2. **Intermediate Layer Adaptation**: Lightweight adapter modules are inserted after the feed-forward network in each transformer block.
3. **Frozen Base Model**: The original pre-trained model parameters remain frozen, ensuring parameter efficiency.

### Adapter Module Design

Each adapter module consists of:
- **Down-projection**: A linear layer that projects the hidden dimension to a smaller bottleneck dimension.
- **Non-linearity**: A GELU activation function.
- **Up-projection**: A linear layer that projects back to the original hidden dimension.
- **Residual connection**: The adapter output is added to the original layer output.

Mathematically, for hidden state `h`:

```
adapter_output = h + Up(GELU(Down(h)))
```

Where `Down` and `Up` are learnable projection matrices.

### Training Procedure

1. Initialize prompts using one of the standard initialization methods (random, from embeddings, etc.).
2. Initialize adapter weights with small random values or using Xavier initialization.
3. Freeze all pre-trained model parameters.
4. Train only the prompt embeddings and adapter parameters on the target task.

## Hyperparameters

Key hyperparameters to tune:

- **Prompt length (P)**: Number of prompt tokens (typical range: 5-100).
- **Adapter bottleneck dimension (D)**: Size of the adapter's hidden layer (typical range: 64-256).
- **Adapter placement**: Which transformer layers receive adapters (all layers vs. selective placement).
- **Learning rate**: Separate learning rates can be used for prompts and adapters.

## Expected Benefits

1. **Performance**: May outperform prompt tuning alone on complex tasks that require fine-grained representation adjustments.
2. **Efficiency**: More parameter-efficient than full fine-tuning, with total parameters typically < 1% of the base model.
3. **Flexibility**: Can be tuned to different task complexities by adjusting prompt length and adapter size.

## Implementation Notes

The implementation in `prompt_tuning/experimental/hybrid_prompt_adapter.py` provides:
- `HybridPromptAdapter`: A Flax module that combines prompts with adapter layers.
- `AdapterLayer`: A reusable adapter module that can be inserted into transformer blocks.
- Configuration options for controlling prompt length, adapter dimensions, and placement strategy.

## References

- Lester et al. (2021). "The Power of Scale for Parameter-Efficient Prompt Tuning." [arXiv:2104.08691](https://arxiv.org/abs/2104.08691)
- Houlsby et al. (2019). "Parameter-Efficient Transfer Learning for NLP." [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)
- He et al. (2022). "Towards a Unified View of Parameter-Efficient Transfer Learning." [arXiv:2110.04366](https://arxiv.org/abs/2110.04366)

## Experimental Status

This is an experimental feature. While the theoretical foundation is sound, extensive empirical validation across different tasks and model sizes is ongoing. Users are encouraged to experiment with different hyperparameter configurations and report their findings.

---

**Note**: This technique is implemented as an experimental feature in the prompt-tuning library. See `prompt_tuning/experimental/hybrid_prompt_adapter.py` for the implementation details.
