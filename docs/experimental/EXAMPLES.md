# Experimental Techniques: Usage Examples

This document provides detailed usage examples for each experimental prompt tuning technique. These examples demonstrate how to integrate the techniques into your training pipeline and configure them for different use cases.

## Table of Contents

1. [P-Tuning v2 Examples](#p-tuning-v2-examples)
2. [Multi-Task Prompt Tuning Examples](#multi-task-prompt-tuning-examples)
3. [Dynamic Prompts Examples](#dynamic-prompts-examples)
4. [Visual Prompt Tuning Examples](#visual-prompt-tuning-examples)
5. [Integration with T5X](#integration-with-t5x)

---

## P-Tuning v2 Examples

### Basic Deep Prompt Tuning

```python
import jax
import jax.numpy as jnp
from prompt_tuning.experimental import p_tuning_v2

# Configuration for a 12-layer transformer
config = p_tuning_v2.DeepPromptConfig(
    num_layers=12,
    prompt_length=20,
    embed_dim=768,
    shared_prompts=False,
    use_encoder=False
)

# Create the deep prompt module
deep_prompt_module = p_tuning_v2.create_deep_prompt_module(config)

# Initialize parameters
rng = jax.random.PRNGKey(0)
params = deep_prompt_module.init(rng)

# Generate deep prompts
deep_prompts = deep_prompt_module.apply(params)
print(f"Deep prompts shape: {deep_prompts.shape}")  # [12, 20, 768]

# Expand to batch size
batch_size = 8
batched_prompts = p_tuning_v2.expand_deep_prompts_to_batch(deep_prompts, batch_size)
print(f"Batched prompts shape: {batched_prompts.shape}")  # [12, 8, 20, 768]

# Use in a transformer layer
layer_idx = 5
layer_input = jnp.zeros((batch_size, 128, 768))  # [B, T, H]
prompted_input = p_tuning_v2.prefix_deep_prompt(
    batched_prompts, layer_idx, layer_input)
print(f"Prompted input shape: {prompted_input.shape}")  # [8, 148, 768]
```

### Shared Prompts Across Layers

```python
# Use shared prompts for parameter efficiency
config = p_tuning_v2.DeepPromptConfig(
    num_layers=12,
    prompt_length=20,
    embed_dim=768,
    shared_prompts=True  # Same prompt for all layers
)

deep_prompt_module = p_tuning_v2.create_deep_prompt_module(config)
params = deep_prompt_module.init(rng)
deep_prompts = deep_prompt_module.apply(params)

# Still returns [12, 20, 768] but uses fewer parameters
print(f"Shape: {deep_prompts.shape}")
```

### MLP-Based Prompt Encoder

```python
# Use an MLP to generate layer-specific prompts
config = p_tuning_v2.DeepPromptConfig(
    num_layers=12,
    prompt_length=20,
    embed_dim=768,
    use_encoder=True,  # Use MLP encoder
    encoder_hidden_dim=512
)

deep_prompt_module = p_tuning_v2.create_deep_prompt_module(config)
params = deep_prompt_module.init(rng)
deep_prompts = deep_prompt_module.apply(params)

# The encoder generates layer-specific prompts from a base prompt
print(f"Encoded prompts shape: {deep_prompts.shape}")  # [12, 20, 768]
```

---

## Multi-Task Prompt Tuning Examples

### Basic Multi-Task Setup

```python
from prompt_tuning.experimental import multitask_prompts

# Create multi-task prompt for 5 tasks
mt_prompt = multitask_prompts.MultiTaskPrompt(
    num_tasks=5,
    shared_length=10,
    task_length=10,
    embed_dim=768,
    composition_method='concat'
)

# Initialize
params = mt_prompt.init(rng)

# Generate prompts for different tasks
for task_id in range(5):
    task_prompt = mt_prompt.apply(params, task_id=task_id)
    print(f"Task {task_id} prompt shape: {task_prompt.shape}")  # [20, 768]
```

### Different Composition Methods

```python
# Concatenation (default)
mt_concat = multitask_prompts.MultiTaskPrompt(
    num_tasks=3,
    shared_length=10,
    task_length=10,
    embed_dim=768,
    composition_method='concat'
)
# Output length: 10 + 10 = 20

# Addition (requires same length)
mt_add = multitask_prompts.MultiTaskPrompt(
    num_tasks=3,
    shared_length=15,
    task_length=15,
    embed_dim=768,
    composition_method='add'
)
# Output length: 15

# Weighted combination
mt_weighted = multitask_prompts.MultiTaskPrompt(
    num_tasks=3,
    shared_length=15,
    task_length=15,
    embed_dim=768,
    composition_method='weighted'
)
# Learns a mixing weight per task

# Gated combination
mt_gated = multitask_prompts.MultiTaskPrompt(
    num_tasks=3,
    shared_length=15,
    task_length=15,
    embed_dim=768,
    composition_method='gated'
)
# Uses an MLP to compute gate weights
```

### Hierarchical Multi-Task Prompts

```python
# Organize tasks into groups
hierarchical_prompt = multitask_prompts.HierarchicalMultiTaskPrompt(
    num_groups=3,
    tasks_per_group=[2, 3, 2],  # Group 0: 2 tasks, Group 1: 3 tasks, Group 2: 2 tasks
    global_length=5,
    group_length=5,
    task_length=5,
    embed_dim=768
)

params = hierarchical_prompt.init(rng)

# Generate prompt for task 1 in group 0
prompt = hierarchical_prompt.apply(params, group_id=0, task_id=1)
print(f"Hierarchical prompt shape: {prompt.shape}")  # [15, 768]
```

### Adaptive Multi-Task Prompts

```python
# Attention-based composition
adaptive_prompt = multitask_prompts.AdaptiveMultiTaskPrompt(
    num_tasks=5,
    num_components=20,
    component_length=10,
    embed_dim=768,
    num_heads=4
)

params = adaptive_prompt.init(rng)

# Generate adaptive prompt for task 2
prompt = adaptive_prompt.apply(params, task_id=2)
print(f"Adaptive prompt shape: {prompt.shape}")  # [10, 768]
```

---

## Dynamic Prompts Examples

### Input-Conditioned Prompts

```python
from prompt_tuning.experimental import dynamic_prompts

# Create input-conditioned prompt
dynamic_prompt = dynamic_prompts.InputConditionedPrompt(
    prompt_length=20,
    embed_dim=768,
    hidden_dim=256,
    num_layers=2,
    pooling_method='mean'
)

params = dynamic_prompt.init(rng, jnp.zeros((1, 128, 768)))

# Generate prompts based on input
batch_size = 8
seq_length = 128
x_embed = jax.random.normal(rng, (batch_size, seq_length, 768))

prompts = dynamic_prompt.apply(params, x_embed)
print(f"Dynamic prompts shape: {prompts.shape}")  # [8, 20, 768]
# Note: Each example in the batch gets a different prompt
```

### Prototype-Based Prompts

```python
# Hard selection (pick one prototype)
prototype_hard = dynamic_prompts.PrototypePrompt(
    num_prototypes=10,
    prompt_length=20,
    embed_dim=768,
    selection_method='hard'
)

params = prototype_hard.init(rng, jnp.zeros((1, 128, 768)))
prompts = prototype_hard.apply(params, x_embed)
print(f"Hard selection prompts: {prompts.shape}")  # [8, 20, 768]

# Soft selection (weighted combination of all prototypes)
prototype_soft = dynamic_prompts.PrototypePrompt(
    num_prototypes=10,
    prompt_length=20,
    embed_dim=768,
    selection_method='soft'
)

params = prototype_soft.init(rng, jnp.zeros((1, 128, 768)))
prompts = prototype_soft.apply(params, x_embed)
print(f"Soft selection prompts: {prompts.shape}")  # [8, 20, 768]

# Top-k selection (combine k most similar prototypes)
prototype_topk = dynamic_prompts.PrototypePrompt(
    num_prototypes=10,
    prompt_length=20,
    embed_dim=768,
    selection_method='top_k',
    top_k=3
)

params = prototype_topk.init(rng, jnp.zeros((1, 128, 768)))
prompts = prototype_topk.apply(params, x_embed)
print(f"Top-k selection prompts: {prompts.shape}")  # [8, 20, 768]
```

### Adaptive Length Prompts

```python
# Prompts with variable length based on input complexity
adaptive_length = dynamic_prompts.AdaptiveLengthPrompt(
    min_length=5,
    max_length=50,
    embed_dim=768,
    complexity_estimator_dim=128
)

params = adaptive_length.init(rng, jnp.zeros((1, 128, 768)))

# Generate prompts with adaptive length
prompts, mask = adaptive_length.apply(params, x_embed)
print(f"Adaptive length prompts: {prompts.shape}")  # [8, 50, 768]
print(f"Mask shape: {mask.shape}")  # [8, 50]
# The mask indicates which prompt positions are active
```

### Context-Aware Prompts

```python
# Prompts that attend to input context
context_aware = dynamic_prompts.ContextAwarePrompt(
    prompt_length=20,
    embed_dim=768,
    num_heads=4
)

params = context_aware.init(rng, jnp.zeros((1, 128, 768)))

# Generate context-aware prompts
prompts = context_aware.apply(params, x_embed)
print(f"Context-aware prompts: {prompts.shape}")  # [8, 20, 768]
# Prompts are adapted via cross-attention with the input
```

---

## Visual Prompt Tuning Examples

### Pixel-Space Prompts

```python
from prompt_tuning.experimental import visual_prompts

# Border prompts
border_prompt = visual_prompts.PixelSpacePrompt(
    image_size=(224, 224),
    prompt_size=10,
    num_channels=3,
    prompt_location='border'
)

params = border_prompt.init(rng, jnp.zeros((1, 224, 224, 3)))

# Add border prompts to images
batch_size = 8
images = jax.random.normal(rng, (batch_size, 224, 224, 3))
prompted_images = border_prompt.apply(params, images)
print(f"Prompted images shape: {prompted_images.shape}")  # [8, 244, 244, 3]

# Corner prompts
corner_prompt = visual_prompts.PixelSpacePrompt(
    image_size=(224, 224),
    prompt_size=20,
    num_channels=3,
    prompt_location='corners'
)

params = corner_prompt.init(rng, jnp.zeros((1, 224, 224, 3)))
prompted_images = corner_prompt.apply(params, images)
print(f"Corner prompted images: {prompted_images.shape}")  # [8, 224, 224, 3]
```

### Patch-Level Prompts for ViT

```python
# Prefix prompts
patch_prompt = visual_prompts.PatchPrompt(
    num_prompts=10,
    patch_dim=768,
    prompt_location='prefix'
)

params = patch_prompt.init(rng, jnp.zeros((1, 196, 768)))

# Add prompts to patch embeddings
num_patches = 196  # For 224x224 image with 16x16 patches
patch_embeddings = jax.random.normal(rng, (batch_size, num_patches, 768))
prompted_patches = patch_prompt.apply(params, patch_embeddings)
print(f"Prompted patches: {prompted_patches.shape}")  # [8, 206, 768]
```

### Deep Visual Prompts

```python
# Deep prompts for multiple ViT layers
deep_visual = visual_prompts.DeepPatchPrompt(
    num_layers=12,
    num_prompts=10,
    patch_dim=768,
    shared_prompts=False
)

params = deep_visual.init(rng)

# Generate deep visual prompts
deep_prompts = deep_visual.apply(params)
print(f"Deep visual prompts: {deep_prompts.shape}")  # [12, 10, 768]
```

### Vision-Language Prompts

```python
# Coordinated prompts for CLIP-like models
vl_prompt = visual_prompts.VisionLanguagePrompt(
    num_text_prompts=10,
    num_visual_prompts=10,
    text_dim=512,
    visual_dim=768,
    shared_projection=True
)

params = vl_prompt.init(rng)

# Generate coordinated prompts
text_prompts, visual_prompts_out = vl_prompt.apply(params)
print(f"Text prompts: {text_prompts.shape}")  # [10, 512]
print(f"Visual prompts: {visual_prompts_out.shape}")  # [10, 768]
```

### Adaptive Visual Prompts

```python
# Visual prompts that adapt to image content
adaptive_visual = visual_prompts.AdaptiveVisualPrompt(
    num_prompts=10,
    patch_dim=768,
    hidden_dim=256
)

params = adaptive_visual.init(rng, jnp.zeros((1, 196, 768)))

# Generate adaptive prompts
adapted_prompts = adaptive_visual.apply(params, patch_embeddings)
print(f"Adaptive visual prompts: {adapted_prompts.shape}")  # [8, 10, 768]
```

---

## Integration with T5X

### Using Deep Prompts in T5X Models

```python
# Example integration with T5X training
from t5x import models
from prompt_tuning.experimental import p_tuning_v2

class DeepPromptT5Model(models.EncoderDecoderModel):
    """T5 model with deep prompt tuning."""
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deep_prompt_config = config
        
    def setup(self):
        super().setup()
        self.deep_prompt = p_tuning_v2.create_deep_prompt_module(
            self.deep_prompt_config)
    
    def encode(self, encoder_input_tokens, encoder_segment_ids=None):
        # Generate deep prompts
        deep_prompts = self.deep_prompt()
        batch_size = encoder_input_tokens.shape[0]
        batched_prompts = p_tuning_v2.expand_deep_prompts_to_batch(
            deep_prompts, batch_size)
        
        # Pass prompts to encoder layers
        # (Implementation depends on T5X encoder structure)
        return super().encode(
            encoder_input_tokens,
            encoder_segment_ids,
            layer_prompts=batched_prompts)
```

### Gin Configuration for Experimental Techniques

```gin
# P-Tuning v2 configuration
from __gin__ import dynamic_registration
from prompt_tuning.experimental import p_tuning_v2

# Deep prompt configuration
p_tuning_v2.DeepPromptConfig:
  num_layers = 12
  prompt_length = 20
  embed_dim = 768
  shared_prompts = False
  use_encoder = False

# Multi-task prompt configuration
from prompt_tuning.experimental import multitask_prompts

multitask_prompts.MultiTaskPrompt:
  num_tasks = 5
  shared_length = 10
  task_length = 10
  embed_dim = 768
  composition_method = 'concat'
```

---

## Performance Tips

### Memory Optimization

```python
# Use shared prompts to reduce memory
config = p_tuning_v2.DeepPromptConfig(
    num_layers=24,  # Large model
    prompt_length=20,
    embed_dim=1024,
    shared_prompts=True  # Reduces parameters significantly
)

# Use prototype selection instead of generation
prototype_prompt = dynamic_prompts.PrototypePrompt(
    num_prototypes=10,  # Small number of prototypes
    prompt_length=20,
    embed_dim=768,
    selection_method='hard'  # Hard selection is faster than soft
)
```

### Gradient Checkpointing

```python
# Use gradient checkpointing for deep prompts
from flax.linen import remat

deep_prompt_module = remat(
    p_tuning_v2.create_deep_prompt_module(config),
    prevent_cse=False
)
```

---

## Troubleshooting

### Common Issues

**Issue**: Out of memory with deep prompts
**Solution**: Use shared prompts or reduce prompt length

**Issue**: Multi-task prompts not learning task-specific features
**Solution**: Try different composition methods or increase task_length

**Issue**: Dynamic prompts not adapting to inputs
**Solution**: Increase hidden_dim or num_layers in the generator network

**Issue**: Visual prompts degrading image quality
**Solution**: Reduce prompt_size or use corner prompts instead of border prompts

---

## Next Steps

- Explore combining multiple techniques (e.g., deep + dynamic prompts)
- Experiment with different hyperparameters for your specific tasks
- Benchmark against standard prompt tuning on your datasets
- Contribute improvements and new techniques back to the repository

For more information, see the main [README](README.md) and individual technique documentation.
