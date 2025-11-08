# Experimental Prompt Tuning Techniques

This directory contains documentation for experimental prompt tuning techniques that extend the core functionality of the prompt-tuning library. These implementations represent cutting-edge research in parameter-efficient fine-tuning and are designed to be modular, extensible, and compatible with the existing T5X/Flaxformer infrastructure.

## Overview

The experimental module includes both **researched techniques** (implementations of published methods) and **novel ideas** (original contributions that combine or extend existing approaches).

## Researched Techniques

These are implementations of techniques from published research papers, adapted to work with the T5X/Flaxformer framework.

### 1. P-Tuning v2 (Deep Prompt Tuning)

**Implementation**: `prompt_tuning/experimental/p_tuning_v2.py`

P-Tuning v2 extends standard prompt tuning by adding trainable prompts at every layer of the transformer, not just the input layer. This provides more fine-grained control over intermediate representations and has been shown to match or exceed full fine-tuning performance on many tasks.

**Key Features**:
- Layer-specific prompts for all transformer layers
- Optional shared prompts across layers for parameter efficiency
- MLP-based prompt encoder for generating layer-specific prompts
- Compatible with both encoder-decoder and decoder-only models

**When to Use**:
- Complex tasks requiring fine-grained representation control
- When standard prompt tuning underperforms
- When you have sufficient compute for training additional parameters

**Reference**: Liu et al. (2022). "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks." [arXiv:2110.07602](https://arxiv.org/abs/2110.07602)

---

### 2. Multi-Task Prompt Tuning

**Implementation**: `prompt_tuning/experimental/multitask_prompts.py`

Multi-task prompt tuning enables efficient learning across multiple related tasks by decomposing prompts into shared and task-specific components. This approach facilitates knowledge transfer between tasks while maintaining task-specific adaptations.

**Key Features**:
- Shared prompt components across all tasks
- Task-specific prompt components for individual tasks
- Multiple composition methods: concatenation, addition, weighted, gated
- Hierarchical organization for task groups
- Attention-based adaptive composition

**When to Use**:
- Training on multiple related tasks simultaneously
- Transfer learning between similar tasks
- Low-resource scenarios where task-specific data is limited

**Reference**: Wang et al. (2022). "Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning." [arXiv:2303.02861](https://arxiv.org/abs/2303.02861)

---

### 3. Dynamic Prompts

**Implementation**: `prompt_tuning/experimental/dynamic_prompts.py`

Dynamic prompts adapt based on input characteristics, unlike static prompts that remain the same for all inputs. This allows the model to use different prompting strategies for different types of inputs.

**Key Features**:
- Input-conditioned prompt generation using MLPs
- Prototype-based prompt selection with soft/hard/top-k methods
- Adaptive prompt length based on input complexity
- Context-aware prompts using cross-attention

**When to Use**:
- Datasets with high input diversity
- Tasks where different inputs require different reasoning strategies
- When you want prompts to adapt to input difficulty

**References**:
- Li et al. (2023). "Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning."
- Vu et al. (2022). "SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer."

---

### 4. Visual Prompt Tuning

**Implementation**: `prompt_tuning/experimental/visual_prompts.py`

Visual prompt tuning extends prompt tuning to vision and vision-language models, enabling parameter-efficient adaptation of visual encoders and multimodal models.

**Key Features**:
- Pixel-space prompts (borders, corners, patches)
- Patch-level prompts for Vision Transformers
- Deep visual prompts for multiple ViT layers
- Coordinated prompts for vision-language models (e.g., CLIP)
- Adaptive visual prompts conditioned on image content

**When to Use**:
- Adapting vision models to new visual domains
- Fine-tuning vision-language models like CLIP
- Image classification, object detection, or visual reasoning tasks

**References**:
- Jia et al. (2022). "Visual Prompt Tuning." [arXiv:2203.12119](https://arxiv.org/abs/2203.12119)
- Zhou et al. (2022). "Learning to Prompt for Vision-Language Models." [arXiv:2109.01134](https://arxiv.org/abs/2109.01134)

---

## Novel Ideas

These are original contributions that combine or extend existing techniques in novel ways. They are experimental and require further validation.

### 1. Hybrid Prompt-Adapter Tuning

**Documentation**: [hybrid_prompt_adapter.md](hybrid_prompt_adapter.md)

Combines prompt tuning with adapter modules to provide adaptation at both the input level (prompts) and intermediate layers (adapters). This hybrid approach aims to achieve better performance than either method alone.

**Key Concepts**:
- Input-level adaptation via prompts
- Intermediate-layer adaptation via lightweight adapters
- Flexible hyperparameter tuning for different task complexities
- More parameter-efficient than full fine-tuning

---

### 2. RL-Prompt Optimization

**Documentation**: [rl_prompt_optimization.md](rl_prompt_optimization.md)

Uses reinforcement learning to automatically discover optimal prompts by treating prompt generation as a sequential decision-making problem. This enables exploration of the prompt space beyond gradient-based optimization.

**Key Concepts**:
- Policy network for generating prompts
- PPO-based training algorithm
- Support for discrete tokens and continuous embeddings
- Flexible reward functions for complex objectives

---

### 3. Cross-Modal Prompting

**Documentation**: [cross_modal_prompting.md](cross_modal_prompting.md)

Enables knowledge transfer between different modalities (text, images, audio) through shared prompt representations. Prompts learned on one modality can improve performance on related tasks in another modality.

**Key Concepts**:
- Shared prompt space across modalities
- Modality-specific projections
- Contrastive alignment loss
- Transfer learning from data-rich to data-scarce modalities

---

## Installation and Setup

The experimental features are part of the main prompt-tuning library. No additional installation is required beyond the standard setup:

```bash
# Clone the repository
git clone https://github.com/google-research/prompt-tuning.git
cd prompt-tuning

# Install dependencies
pip install -r requirements.txt

# Install the library in development mode
pip install -e .
```

## Usage Examples

### P-Tuning v2

```python
from prompt_tuning.experimental import p_tuning_v2

# Create deep prompt configuration
config = p_tuning_v2.DeepPromptConfig(
    num_layers=12,
    prompt_length=20,
    embed_dim=768,
    shared_prompts=False,
    use_encoder=False
)

# Create deep prompt module
deep_prompt = p_tuning_v2.create_deep_prompt_module(config)

# Generate prompts for all layers
layer_prompts = deep_prompt()  # Shape: [12, 20, 768]

# Expand to batch size
batch_prompts = p_tuning_v2.expand_deep_prompts_to_batch(layer_prompts, batch_size=32)
```

### Multi-Task Prompt Tuning

```python
from prompt_tuning.experimental import multitask_prompts

# Create multi-task prompt
mt_prompt = multitask_prompts.MultiTaskPrompt(
    num_tasks=5,
    shared_length=10,
    task_length=10,
    embed_dim=768,
    composition_method='concat'
)

# Generate prompt for task 0
task_0_prompt = mt_prompt(task_id=0)  # Shape: [20, 768]

# Generate prompt for task 3
task_3_prompt = mt_prompt(task_id=3)  # Shape: [20, 768]
```

### Dynamic Prompts

```python
from prompt_tuning.experimental import dynamic_prompts

# Input-conditioned prompts
dynamic_prompt = dynamic_prompts.InputConditionedPrompt(
    prompt_length=20,
    embed_dim=768,
    hidden_dim=256,
    pooling_method='mean'
)

# Generate prompts based on input
prompts = dynamic_prompt(x_embed)  # x_embed: [B, T, H] -> prompts: [B, P, H]

# Prototype-based prompts
prototype_prompt = dynamic_prompts.PrototypePrompt(
    num_prototypes=10,
    prompt_length=20,
    embed_dim=768,
    selection_method='soft'
)

# Select prompts based on input similarity
selected_prompts = prototype_prompt(x_embed)  # [B, P, H]
```

### Visual Prompt Tuning

```python
from prompt_tuning.experimental import visual_prompts

# Patch-level prompts for ViT
patch_prompt = visual_prompts.PatchPrompt(
    num_prompts=10,
    patch_dim=768,
    prompt_location='prefix'
)

# Add prompts to patch embeddings
prompted_patches = patch_prompt(patch_embeddings)  # [B, N, D] -> [B, N+10, D]

# Vision-language prompts
vl_prompt = visual_prompts.VisionLanguagePrompt(
    num_text_prompts=10,
    num_visual_prompts=10,
    text_dim=512,
    visual_dim=768,
    shared_projection=True
)

# Generate coordinated prompts
text_prompts, visual_prompts = vl_prompt()
```

## Testing

Each experimental module includes comprehensive unit tests. To run tests:

```bash
# Run all experimental tests
python -m pytest prompt_tuning/experimental/

# Run tests for a specific module
python -m pytest prompt_tuning/experimental/test_p_tuning_v2.py
```

## Contributing

We welcome contributions to the experimental module. When adding new techniques:

1. Follow the existing code structure and style
2. Include comprehensive docstrings (Google style)
3. Add unit tests with good coverage
4. Create documentation in `docs/experimental/`
5. Update this README with your technique

## Citation

If you use these experimental techniques in your research, please cite both the original papers (referenced in each technique's documentation) and this repository:

```bibtex
@software{prompt_tuning_experimental,
  title = {Experimental Prompt Tuning Techniques},
  author = {Google Research},
  year = {2024},
  url = {https://github.com/google-research/prompt-tuning}
}
```

## License

All code in this directory is licensed under the Apache License 2.0, consistent with the main prompt-tuning repository.

## Status and Roadmap

**Current Status**: All four researched techniques are implemented and ready for experimentation. The three novel ideas are documented but implementations are in progress.

**Roadmap**:
- [ ] Implement Hybrid Prompt-Adapter Tuning
- [ ] Implement RL-Prompt Optimization
- [ ] Implement Cross-Modal Prompting
- [ ] Add comprehensive benchmarks across multiple tasks
- [ ] Create example notebooks for each technique
- [ ] Integrate with T5X training pipeline
- [ ] Add support for model parallelism and large-scale training

## Support

For questions, issues, or feature requests related to experimental techniques:

1. Check the documentation in this directory
2. Search existing GitHub issues
3. Open a new issue with the `experimental` label

---

**Note**: These are experimental features under active development. APIs may change, and some techniques may require further validation before production use.
