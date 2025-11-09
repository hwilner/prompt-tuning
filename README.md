# Prompt Tuning: Advanced Techniques Fork

**This is a community-enhanced fork of the official [google-research/prompt-tuning](https://github.com/google-research/prompt-tuning) repository.**

This fork includes implementations of several advanced prompt tuning techniques from recent research, along with comprehensive documentation and examples. The goal of this fork is to provide a platform for researchers and practitioners to experiment with cutting-edge, parameter-efficient fine-tuning methods.

---

## What's New in This Fork

This fork introduces four advanced prompt tuning techniques:

1.  **P-Tuning v2 (Deep Prompt Tuning)**: Adds trainable prompts at every layer of the transformer, enabling more fine-grained control over intermediate representations.
2.  **Multi-Task Prompt Tuning**: Enables efficient learning across multiple related tasks by decomposing prompts into shared and task-specific components.
3.  **Dynamic Prompts**: Prompts that adapt based on input characteristics, allowing the model to use different prompting strategies for different types of inputs.
4.  **Visual Prompt Tuning**: Extends prompt tuning to vision and vision-language models, enabling parameter-efficient adaptation of visual encoders and multimodal models.

In addition, this fork includes detailed documentation for three novel, experimental ideas:

*   **Hybrid Prompt-Adapter Tuning**: Combines prompt tuning with adapter modules for multi-level adaptation.
*   **RL-Prompt Optimization**: Uses reinforcement learning to discover optimal prompts.
*   **Cross-Modal Prompting**: Enables knowledge transfer between modalities through shared prompt representations.

For more details, see the [experimental documentation](docs/experimental/README.md).

---

## How to Use the Advanced Techniques

All new techniques are located in the `prompt_tuning/experimental` directory. Here's a quick example of how to use P-Tuning v2:

```python
from prompt_tuning.experimental import p_tuning_v2

# Create deep prompt configuration
config = p_tuning_v2.DeepPromptConfig(
    num_layers=12,
    prompt_length=20,
    embed_dim=768,
    shared_prompts=False
)

# Create and use deep prompt module
deep_prompt = p_tuning_v2.create_deep_prompt_module(config)
layer_prompts = deep_prompt()  # Shape: [12, 20, 768]
```

For more detailed examples, please see the [examples documentation](docs/experimental/EXAMPLES.md).

---

## Original README

(The original README from the `google-research/prompt-tuning` repository is included below for reference.)

---


original_readme_content
