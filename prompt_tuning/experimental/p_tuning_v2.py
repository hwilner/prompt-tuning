# Copyright 2024 Google.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""P-Tuning v2 (Deep Prompt Tuning) implementation.

P-Tuning v2 extends standard prompt tuning by adding trainable prompts at every
layer of the transformer, not just the input layer. This allows for more
fine-grained control over the model's intermediate representations.

Reference:
  Liu et al. (2022). "P-Tuning v2: Prompt Tuning Can Be Comparable to
  Fine-tuning Universally Across Scales and Tasks."
  https://arxiv.org/abs/2110.07602
"""

from typing import Callable, Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp
from prompt_tuning import prompts
from flaxformer.types import Array, DType


class DeepPrompt(nn.Module):
  """Deep prompt tuning module that creates prompts for multiple layers.
  
  Attributes:
    num_layers: Number of transformer layers to add prompts to.
    prompt_length: Length of the prompt at each layer.
    embed_dim: Embedding dimension of the model.
    prompt_init: Initializer for prompt embeddings.
    dtype: Data type for prompt parameters.
    shared_prompts: If True, use the same prompt across all layers.
  """
  num_layers: int
  prompt_length: int
  embed_dim: int
  prompt_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.uniform()
  dtype: DType = jnp.float32
  shared_prompts: bool = False

  @nn.compact
  def __call__(self) -> Array:
    """Generate deep prompts for all layers.
    
    Returns:
      Deep prompts with shape [num_layers, prompt_length, embed_dim] if not
      shared, or [prompt_length, embed_dim] if shared.
    """
    if self.shared_prompts:
      # Single prompt shared across all layers
      prompt = self.param(
          'shared_prompt',
          self.prompt_init,
          (self.prompt_length, self.embed_dim),
          self.dtype)
      # Expand to all layers
      return jnp.tile(
          jnp.expand_dims(prompt, axis=0),
          [self.num_layers, 1, 1])
    else:
      # Separate prompt for each layer
      return self.param(
          'layer_prompts',
          self.prompt_init,
          (self.num_layers, self.prompt_length, self.embed_dim),
          self.dtype)


class DeepPromptEncoder(nn.Module):
  """Encoder that generates deep prompts using a small MLP.
  
  This variant uses a small neural network to generate layer-specific prompts
  from a base prompt, which can be more parameter-efficient than storing
  separate prompts for each layer.
  
  Attributes:
    num_layers: Number of transformer layers.
    prompt_length: Length of the prompt.
    embed_dim: Embedding dimension.
    hidden_dim: Hidden dimension of the encoder MLP.
    base_prompt_init: Initializer for the base prompt.
    dtype: Data type for parameters.
  """
  num_layers: int
  prompt_length: int
  embed_dim: int
  hidden_dim: int = 512
  base_prompt_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.uniform()
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self) -> Array:
    """Generate deep prompts using an MLP encoder.
    
    Returns:
      Deep prompts with shape [num_layers, prompt_length, embed_dim].
    """
    # Base prompt that is shared
    base_prompt = self.param(
        'base_prompt',
        self.base_prompt_init,
        (self.prompt_length, self.embed_dim),
        self.dtype)
    
    # Layer embeddings to condition the encoder
    layer_embeddings = self.param(
        'layer_embeddings',
        nn.initializers.normal(stddev=0.02),
        (self.num_layers, self.hidden_dim),
        self.dtype)
    
    # MLP to generate layer-specific prompts
    # For each layer, we concatenate the base prompt with the layer embedding
    # and pass through an MLP to get the layer-specific prompt
    layer_prompts = []
    for layer_idx in range(self.num_layers):
      # Get layer embedding
      layer_emb = layer_embeddings[layer_idx]  # [hidden_dim]
      
      # Expand to match prompt length
      layer_emb_expanded = jnp.tile(
          jnp.expand_dims(layer_emb, axis=0),
          [self.prompt_length, 1])  # [prompt_length, hidden_dim]
      
      # Concatenate with base prompt
      combined = jnp.concatenate(
          [base_prompt, layer_emb_expanded],
          axis=-1)  # [prompt_length, embed_dim + hidden_dim]
      
      # Pass through MLP
      hidden = nn.Dense(
          features=self.hidden_dim,
          dtype=self.dtype,
          name=f'encoder_hidden_{layer_idx}')(combined)
      hidden = nn.gelu(hidden)
      
      layer_prompt = nn.Dense(
          features=self.embed_dim,
          dtype=self.dtype,
          name=f'encoder_output_{layer_idx}')(hidden)
      
      layer_prompts.append(layer_prompt)
    
    return jnp.stack(layer_prompts, axis=0)  # [num_layers, prompt_length, embed_dim]


def expand_deep_prompts_to_batch(deep_prompts: Array, batch_size: int) -> Array:
  """Expand unbatched deep prompts to batch size.
  
  Args:
    deep_prompts: Deep prompts with shape [num_layers, prompt_length, embed_dim].
    batch_size: Target batch size.
    
  Returns:
    Batched deep prompts with shape [num_layers, batch_size, prompt_length, embed_dim].
  """
  return jnp.tile(
      jnp.expand_dims(deep_prompts, axis=1),
      [1, batch_size, 1, 1])


def prefix_deep_prompt(
    deep_prompts: Array,
    layer_idx: int,
    x_embed: Array) -> Array:
  """Concatenate layer-specific prompt to the beginning of layer input.
  
  Args:
    deep_prompts: Deep prompts with shape [num_layers, B, P, H].
    layer_idx: Index of the current layer.
    x_embed: Layer input with shape [B, T, H].
    
  Returns:
    Input with prompt concatenated to the front with shape [B, P + T, H].
  """
  layer_prompt = deep_prompts[layer_idx]  # [B, P, H]
  return jnp.concatenate([layer_prompt, x_embed], axis=1)


class DeepPromptConfig:
  """Configuration for deep prompt tuning.
  
  Attributes:
    num_layers: Number of transformer layers.
    prompt_length: Length of prompts at each layer.
    embed_dim: Embedding dimension.
    shared_prompts: Whether to share prompts across layers.
    use_encoder: Whether to use MLP encoder for generating prompts.
    encoder_hidden_dim: Hidden dimension for encoder MLP.
    prompt_init: Initialization method for prompts.
  """
  
  def __init__(
      self,
      num_layers: int,
      prompt_length: int,
      embed_dim: int,
      shared_prompts: bool = False,
      use_encoder: bool = False,
      encoder_hidden_dim: int = 512,
      prompt_init: Optional[Callable[[Array, Sequence[int]], Array]] = None):
    self.num_layers = num_layers
    self.prompt_length = prompt_length
    self.embed_dim = embed_dim
    self.shared_prompts = shared_prompts
    self.use_encoder = use_encoder
    self.encoder_hidden_dim = encoder_hidden_dim
    self.prompt_init = prompt_init or nn.initializers.uniform()


def create_deep_prompt_module(config: DeepPromptConfig) -> nn.Module:
  """Create a deep prompt module based on configuration.
  
  Args:
    config: Deep prompt configuration.
    
  Returns:
    A DeepPrompt or DeepPromptEncoder module.
  """
  if config.use_encoder:
    return DeepPromptEncoder(
        num_layers=config.num_layers,
        prompt_length=config.prompt_length,
        embed_dim=config.embed_dim,
        hidden_dim=config.encoder_hidden_dim,
        base_prompt_init=config.prompt_init)
  else:
    return DeepPrompt(
        num_layers=config.num_layers,
        prompt_length=config.prompt_length,
        embed_dim=config.embed_dim,
        prompt_init=config.prompt_init,
        shared_prompts=config.shared_prompts)
