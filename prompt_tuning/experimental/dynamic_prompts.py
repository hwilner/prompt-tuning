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

"""Dynamic Prompt Tuning with input-adaptive prompts.

This module implements dynamic prompts that adapt based on input characteristics.
Unlike static prompts that are the same for all inputs, dynamic prompts are
generated or selected based on properties of the input sequence.

References:
  - Li et al. (2023). "Dynamic Prompt Learning via Policy Gradient for
    Semi-structured Mathematical Reasoning."
  - Vu et al. (2022). "SPoT: Better Frozen Model Adaptation through Soft Prompt
    Transfer."
"""

from typing import Callable, Optional, Sequence
import flax.linen as nn
import jax
import jax.numpy as jnp
from flaxformer.types import Array, DType


class InputConditionedPrompt(nn.Module):
  """Generate prompts conditioned on input embeddings.
  
  This module uses a small neural network to generate prompts based on
  the input sequence embeddings, allowing the prompt to adapt to different
  types of inputs.
  
  Attributes:
    prompt_length: Length of the generated prompt.
    embed_dim: Embedding dimension.
    hidden_dim: Hidden dimension of the generator network.
    num_layers: Number of layers in the generator network.
    pooling_method: How to pool input embeddings ('mean', 'max', 'first', 'last').
    dtype: Data type for parameters.
  """
  prompt_length: int
  embed_dim: int
  hidden_dim: int = 256
  num_layers: int = 2
  pooling_method: str = 'mean'
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x_embed: Array) -> Array:
    """Generate prompt conditioned on input.
    
    Args:
      x_embed: Input embeddings with shape [B, T, H].
      
    Returns:
      Generated prompts with shape [B, P, H].
    """
    batch_size = x_embed.shape[0]
    
    # Pool input embeddings to get a fixed-size representation
    if self.pooling_method == 'mean':
      pooled = jnp.mean(x_embed, axis=1)  # [B, H]
    elif self.pooling_method == 'max':
      pooled = jnp.max(x_embed, axis=1)  # [B, H]
    elif self.pooling_method == 'first':
      pooled = x_embed[:, 0, :]  # [B, H]
    elif self.pooling_method == 'last':
      pooled = x_embed[:, -1, :]  # [B, H]
    else:
      raise ValueError(
          f"Unknown pooling_method: {self.pooling_method}. "
          f"Must be one of: 'mean', 'max', 'first', 'last'")
    
    # Pass through MLP to generate prompt
    hidden = pooled
    for i in range(self.num_layers):
      hidden = nn.Dense(
          features=self.hidden_dim,
          dtype=self.dtype,
          name=f'generator_layer_{i}')(hidden)
      hidden = nn.gelu(hidden)
    
    # Generate prompt embeddings
    # Output shape: [B, P * H]
    prompt_flat = nn.Dense(
        features=self.prompt_length * self.embed_dim,
        dtype=self.dtype,
        name='prompt_output')(hidden)
    
    # Reshape to prompt format
    prompt = jnp.reshape(
        prompt_flat,
        (batch_size, self.prompt_length, self.embed_dim))
    
    return prompt


class PrototypePrompt(nn.Module):
  """Select prompts from a learned set of prototypes based on input.
  
  This module maintains a set of prototype prompts and selects or combines
  them based on similarity to the input. This is more parameter-efficient
  than generating prompts from scratch.
  
  Attributes:
    num_prototypes: Number of prototype prompts.
    prompt_length: Length of each prototype prompt.
    embed_dim: Embedding dimension.
    selection_method: How to select prompts ('hard', 'soft', 'top_k').
    top_k: Number of prototypes to combine (for 'top_k' method).
    prompt_init: Initializer for prototype prompts.
    dtype: Data type for parameters.
  """
  num_prototypes: int
  prompt_length: int
  embed_dim: int
  selection_method: str = 'soft'
  top_k: int = 3
  prompt_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.uniform()
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x_embed: Array) -> Array:
    """Select prompt based on input similarity to prototypes.
    
    Args:
      x_embed: Input embeddings with shape [B, T, H].
      
    Returns:
      Selected or combined prompts with shape [B, P, H].
    """
    batch_size = x_embed.shape[0]
    
    # Prototype prompts
    prototypes = self.param(
        'prototypes',
        self.prompt_init,
        (self.num_prototypes, self.prompt_length, self.embed_dim),
        self.dtype)
    
    # Prototype keys for similarity computation
    prototype_keys = self.param(
        'prototype_keys',
        nn.initializers.normal(stddev=0.02),
        (self.num_prototypes, self.embed_dim),
        self.dtype)
    
    # Compute input representation (mean pooling)
    input_repr = jnp.mean(x_embed, axis=1)  # [B, H]
    
    # Compute similarity between input and prototypes
    # [B, H] @ [H, num_prototypes] -> [B, num_prototypes]
    similarities = jnp.matmul(input_repr, prototype_keys.T)
    similarities = similarities / jnp.sqrt(self.embed_dim)
    
    if self.selection_method == 'hard':
      # Select single most similar prototype
      selected_indices = jnp.argmax(similarities, axis=-1)  # [B]
      selected_prompts = prototypes[selected_indices]  # [B, P, H]
      return selected_prompts
    
    elif self.selection_method == 'soft':
      # Soft combination of all prototypes
      weights = nn.softmax(similarities, axis=-1)  # [B, num_prototypes]
      # [B, num_prototypes, 1, 1] * [num_prototypes, P, H] -> [B, num_prototypes, P, H]
      weighted_prototypes = (
          jnp.expand_dims(jnp.expand_dims(weights, -1), -1) * 
          jnp.expand_dims(prototypes, 0))
      # Sum over prototypes
      combined_prompts = jnp.sum(weighted_prototypes, axis=1)  # [B, P, H]
      return combined_prompts
    
    elif self.selection_method == 'top_k':
      # Combine top-k most similar prototypes
      top_k_indices = jnp.argsort(similarities, axis=-1)[:, -self.top_k:]  # [B, k]
      top_k_similarities = jnp.take_along_axis(
          similarities, top_k_indices, axis=-1)  # [B, k]
      
      # Normalize weights
      weights = nn.softmax(top_k_similarities, axis=-1)  # [B, k]
      
      # Gather top-k prototypes
      # This is a bit tricky in JAX, we need to use advanced indexing
      batch_indices = jnp.arange(batch_size)[:, None]  # [B, 1]
      selected_prototypes = prototypes[top_k_indices]  # [B, k, P, H]
      
      # Weight and combine
      weighted = (
          jnp.expand_dims(jnp.expand_dims(weights, -1), -1) * 
          selected_prototypes)  # [B, k, P, H]
      combined_prompts = jnp.sum(weighted, axis=1)  # [B, P, H]
      return combined_prompts
    
    else:
      raise ValueError(
          f"Unknown selection_method: {self.selection_method}. "
          f"Must be one of: 'hard', 'soft', 'top_k'")


class AdaptiveLengthPrompt(nn.Module):
  """Prompts with adaptive length based on input complexity.
  
  This module generates prompts whose length varies based on input
  characteristics, using more prompt tokens for complex inputs.
  
  Attributes:
    min_length: Minimum prompt length.
    max_length: Maximum prompt length.
    embed_dim: Embedding dimension.
    complexity_estimator_dim: Hidden dimension for complexity estimator.
    prompt_init: Initializer for prompt embeddings.
    dtype: Data type for parameters.
  """
  min_length: int
  max_length: int
  embed_dim: int
  complexity_estimator_dim: int = 128
  prompt_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.uniform()
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x_embed: Array) -> Array:
    """Generate adaptive-length prompt based on input.
    
    Args:
      x_embed: Input embeddings with shape [B, T, H].
      
    Returns:
      Prompts with variable effective length, shape [B, max_length, H].
      A mask is also returned to indicate active prompt positions.
    """
    batch_size = x_embed.shape[0]
    
    # Full-length prompt pool
    prompt_pool = self.param(
        'prompt_pool',
        self.prompt_init,
        (self.max_length, self.embed_dim),
        self.dtype)
    
    # Estimate input complexity
    input_repr = jnp.mean(x_embed, axis=1)  # [B, H]
    
    complexity_hidden = nn.Dense(
        features=self.complexity_estimator_dim,
        dtype=self.dtype,
        name='complexity_hidden')(input_repr)
    complexity_hidden = nn.gelu(complexity_hidden)
    
    # Predict prompt length (as a continuous value)
    length_logit = nn.Dense(
        features=1,
        dtype=self.dtype,
        name='length_predictor')(complexity_hidden)
    
    # Map to [min_length, max_length] range
    normalized_length = nn.sigmoid(length_logit)  # [B, 1]
    predicted_length = (
        self.min_length + 
        normalized_length * (self.max_length - self.min_length))
    
    # Create soft mask based on predicted length
    positions = jnp.arange(self.max_length)  # [max_length]
    # [B, 1] vs [max_length] -> [B, max_length]
    mask = jnp.less(
        jnp.expand_dims(positions, 0),
        predicted_length)
    mask = mask.astype(self.dtype)
    
    # Apply mask to prompt
    # [B, max_length, 1] * [max_length, H] -> [B, max_length, H]
    masked_prompt = (
        jnp.expand_dims(mask, -1) * 
        jnp.expand_dims(prompt_pool, 0))
    
    return masked_prompt, mask


class ContextAwarePrompt(nn.Module):
  """Prompts that attend to input context.
  
  This module uses cross-attention between a base prompt and the input
  to create context-aware prompt representations.
  
  Attributes:
    prompt_length: Length of the base prompt.
    embed_dim: Embedding dimension.
    num_heads: Number of attention heads.
    prompt_init: Initializer for base prompt.
    dtype: Data type for parameters.
  """
  prompt_length: int
  embed_dim: int
  num_heads: int = 4
  prompt_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.uniform()
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, x_embed: Array) -> Array:
    """Generate context-aware prompt via cross-attention.
    
    Args:
      x_embed: Input embeddings with shape [B, T, H].
      
    Returns:
      Context-aware prompts with shape [B, P, H].
    """
    batch_size = x_embed.shape[0]
    
    # Base prompt (queries)
    base_prompt = self.param(
        'base_prompt',
        self.prompt_init,
        (self.prompt_length, self.embed_dim),
        self.dtype)
    
    # Expand to batch
    queries = jnp.tile(
        jnp.expand_dims(base_prompt, 0),
        [batch_size, 1, 1])  # [B, P, H]
    
    # Input as keys and values
    keys = x_embed  # [B, T, H]
    values = x_embed  # [B, T, H]
    
    # Multi-head cross-attention
    head_dim = self.embed_dim // self.num_heads
    
    # Project queries, keys, values
    q = nn.Dense(
        features=self.embed_dim,
        dtype=self.dtype,
        name='query_proj')(queries)
    k = nn.Dense(
        features=self.embed_dim,
        dtype=self.dtype,
        name='key_proj')(keys)
    v = nn.Dense(
        features=self.embed_dim,
        dtype=self.dtype,
        name='value_proj')(values)
    
    # Reshape for multi-head attention
    # [B, P, H] -> [B, P, num_heads, head_dim] -> [B, num_heads, P, head_dim]
    q = jnp.transpose(
        jnp.reshape(q, (batch_size, self.prompt_length, self.num_heads, head_dim)),
        (0, 2, 1, 3))
    # [B, T, H] -> [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
    k = jnp.transpose(
        jnp.reshape(k, (batch_size, -1, self.num_heads, head_dim)),
        (0, 2, 1, 3))
    v = jnp.transpose(
        jnp.reshape(v, (batch_size, -1, self.num_heads, head_dim)),
        (0, 2, 1, 3))
    
    # Attention scores
    # [B, num_heads, P, head_dim] @ [B, num_heads, head_dim, T] -> [B, num_heads, P, T]
    attention_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))
    attention_scores = attention_scores / jnp.sqrt(head_dim)
    attention_weights = nn.softmax(attention_scores, axis=-1)
    
    # Apply attention to values
    # [B, num_heads, P, T] @ [B, num_heads, T, head_dim] -> [B, num_heads, P, head_dim]
    attended = jnp.matmul(attention_weights, v)
    
    # Reshape back
    # [B, num_heads, P, head_dim] -> [B, P, num_heads, head_dim] -> [B, P, H]
    attended = jnp.transpose(attended, (0, 2, 1, 3))
    attended = jnp.reshape(attended, (batch_size, self.prompt_length, self.embed_dim))
    
    # Output projection
    output = nn.Dense(
        features=self.embed_dim,
        dtype=self.dtype,
        name='output_proj')(attended)
    
    # Residual connection with base prompt
    return output + queries
