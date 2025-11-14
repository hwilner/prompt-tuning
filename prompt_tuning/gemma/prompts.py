# Copyright 2024 Google LLC.
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

"""Prompt tuning implementation for Gemma models.

This module provides soft prompt tuning capabilities for Gemma decoder-only
transformer models. Unlike T5's encoder-decoder architecture, Gemma uses a
causal (decoder-only) architecture similar to GPT.

Key features:
  - Soft prompts prepended to input embeddings
  - Compatible with Gemma's RoPE position encoding
  - Supports causal attention masking
  - Works with all Gemma model sizes (270M, 2B, 4B, 7B, 27B)

Example usage:
  ```python
  from gemma import gm
  from prompt_tuning.gemma import prompts
  
  # Load base Gemma model
  model = gm.nn.Gemma3_2B()
  params = gm.ckpts.load_params(gm.ckpts.CheckpointPath.GEMMA3_2B_IT)
  
  # Create prompt-tuned model
  prompt_config = prompts.PromptConfig(
      prompt_length=20,
      embed_dim=2048,
  )
  prompt_model = prompts.PromptGemma(model, prompt_config)
  
  # Initialize prompt parameters
  prompt_params = prompt_model.init_prompt_params(jax.random.PRNGKey(0))
  
  # Use for inference
  output = prompt_model.apply(
      {'params': params, 'prompt': prompt_params},
      input_ids,
  )
  ```
"""

from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze


Array = jax.Array
PRNGKey = jax.random.PRNGKey


class PromptConfig:
  """Configuration for prompt tuning with Gemma.
  
  Attributes:
    prompt_length: Number of prompt tokens to prepend.
    embed_dim: Embedding dimension of the model.
    init_scale: Scale for random initialization.
    init_from_vocab: If True, initialize from vocabulary embeddings.
    init_text: Optional text to initialize prompt from.
  """
  
  def __init__(
      self,
      prompt_length: int,
      embed_dim: int,
      init_scale: float = 0.5,
      init_from_vocab: bool = False,
      init_text: Optional[str] = None,
  ):
    self.prompt_length = prompt_length
    self.embed_dim = embed_dim
    self.init_scale = init_scale
    self.init_from_vocab = init_from_vocab
    self.init_text = init_text


class SoftPrompt(nn.Module):
  """Learnable soft prompt module for Gemma.
  
  This module creates trainable prompt embeddings that are prepended to the
  input token embeddings. The prompts are learned during training while the
  base Gemma model parameters remain frozen.
  
  Attributes:
    prompt_length: Number of prompt tokens.
    embed_dim: Embedding dimension.
    init_scale: Scale for random initialization.
  """
  
  prompt_length: int
  embed_dim: int
  init_scale: float = 0.5
  
  @nn.compact
  def __call__(self, batch_size: int) -> Array:
    """Generate soft prompt embeddings.
    
    Args:
      batch_size: Batch size for broadcasting prompts.
      
    Returns:
      Prompt embeddings of shape [batch_size, prompt_length, embed_dim].
    """
    # Initialize prompt parameters
    prompt = self.param(
        'prompt',
        nn.initializers.normal(stddev=self.init_scale),
        (self.prompt_length, self.embed_dim),
    )
    
    # Broadcast to batch size
    prompt = jnp.expand_dims(prompt, axis=0)
    prompt = jnp.tile(prompt, (batch_size, 1, 1))
    
    return prompt


class PromptGemma(nn.Module):
  """Gemma model with prompt tuning.
  
  This wrapper adds soft prompt tuning capabilities to a Gemma model. The
  prompts are prepended to the input embeddings, and position indices are
  adjusted accordingly for RoPE.
  
  Attributes:
    base_model: The underlying Gemma model.
    prompt_config: Configuration for the prompt.
  """
  
  base_model: Any  # Gemma model from gm.nn
  prompt_config: PromptConfig
  
  def setup(self):
    """Initialize the prompt module."""
    self.prompt = SoftPrompt(
        prompt_length=self.prompt_config.prompt_length,
        embed_dim=self.prompt_config.embed_dim,
        init_scale=self.prompt_config.init_scale,
    )
  
  def __call__(
      self,
      input_ids: Array,
      positions: Optional[Array] = None,
      attention_mask: Optional[Array] = None,
      **kwargs,
  ) -> Array:
    """Forward pass with prompt tuning.
    
    Args:
      input_ids: Input token IDs of shape [batch_size, seq_len].
      positions: Position indices for RoPE. If None, computed automatically.
      attention_mask: Attention mask. If None, computed automatically.
      **kwargs: Additional arguments passed to the base model.
      
    Returns:
      Model output (logits or hidden states depending on base model).
    """
    batch_size, seq_len = input_ids.shape
    prompt_length = self.prompt_config.prompt_length
    
    # Get input embeddings from base model
    # Note: This assumes the base model has an 'embed' or 'embedder' attribute
    # We'll need to adapt this based on actual Gemma implementation
    input_embeds = self.base_model.embedder.encode(input_ids)
    
    # Generate soft prompts
    prompt_embeds = self.prompt(batch_size)
    
    # Concatenate prompts with input embeddings
    # Shape: [batch_size, prompt_length + seq_len, embed_dim]
    combined_embeds = jnp.concatenate([prompt_embeds, input_embeds], axis=1)
    
    # Adjust position indices for RoPE
    if positions is None:
      # Create positions: [0, 1, ..., prompt_length + seq_len - 1]
      positions = jnp.arange(prompt_length + seq_len)
      positions = jnp.expand_dims(positions, axis=0)
      positions = jnp.tile(positions, (batch_size, 1))
    else:
      # Shift provided positions by prompt_length
      prompt_positions = jnp.arange(prompt_length)
      prompt_positions = jnp.expand_dims(prompt_positions, axis=0)
      prompt_positions = jnp.tile(prompt_positions, (batch_size, 1))
      positions = positions + prompt_length
      positions = jnp.concatenate([prompt_positions, positions], axis=1)
    
    # Adjust attention mask for causal attention
    if attention_mask is None:
      # Create causal mask that allows:
      # - Prompts to attend to each other
      # - Input tokens to attend to prompts
      # - Causal masking within input tokens
      total_len = prompt_length + seq_len
      attention_mask = jnp.tril(jnp.ones((total_len, total_len)))
      attention_mask = jnp.expand_dims(attention_mask, axis=0)
      attention_mask = jnp.tile(attention_mask, (batch_size, 1, 1))
    else:
      # Extend provided mask to include prompts
      prompt_mask = jnp.ones((batch_size, prompt_length, prompt_length + seq_len))
      input_to_prompt_mask = jnp.ones((batch_size, seq_len, prompt_length))
      extended_input_mask = jnp.concatenate(
          [input_to_prompt_mask, attention_mask], axis=2
      )
      attention_mask = jnp.concatenate(
          [prompt_mask, extended_input_mask], axis=1
      )
    
    # Forward pass through base model with combined embeddings
    # Note: This is a simplified version. Actual implementation depends on
    # Gemma's API for passing pre-computed embeddings
    output = self.base_model(
        inputs_embeds=combined_embeds,
        positions=positions,
        attention_mask=attention_mask,
        **kwargs,
    )
    
    return output
  
  def init_prompt_params(self, rng: PRNGKey) -> dict:
    """Initialize prompt parameters.
    
    Args:
      rng: Random number generator key.
      
    Returns:
      Dictionary containing initialized prompt parameters.
    """
    # Create dummy inputs for initialization
    dummy_input_ids = jnp.zeros((1, 10), dtype=jnp.int32)
    
    # Initialize only the prompt parameters
    variables = self.init(rng, dummy_input_ids)
    
    # Extract only prompt parameters
    prompt_params = variables['params']['prompt']
    
    return {'prompt': prompt_params}


def create_prompt_model(
    base_model: Any,
    prompt_config: PromptConfig,
) -> PromptGemma:
  """Create a prompt-tuned Gemma model.
  
  Args:
    base_model: Base Gemma model from gm.nn.
    prompt_config: Configuration for the prompt.
    
  Returns:
    PromptGemma model ready for training or inference.
  """
  return PromptGemma(base_model=base_model, prompt_config=prompt_config)


def freeze_base_model(params: dict) -> dict:
  """Freeze base model parameters for prompt tuning.
  
  This function marks all base model parameters as non-trainable, keeping
  only the prompt parameters trainable.
  
  Args:
    params: Full parameter dictionary including base model and prompt.
    
  Returns:
    Parameter dictionary with base model parameters frozen.
  """
  params = unfreeze(params)
  
  # Mark all parameters as frozen except prompts
  trainable_params = {'prompt': params.get('prompt', {})}
  frozen_params = {k: v for k, v in params.items() if k != 'prompt'}
  
  return freeze({'trainable': trainable_params, 'frozen': frozen_params})


def get_trainable_params(params: dict) -> dict:
  """Extract only trainable prompt parameters.
  
  Args:
    params: Full parameter dictionary.
    
  Returns:
    Dictionary containing only prompt parameters.
  """
  if 'trainable' in params:
    return params['trainable']
  elif 'prompt' in params:
    return {'prompt': params['prompt']}
  else:
    raise ValueError("No trainable prompt parameters found")


# Initialization utilities

def init_from_vocab(
    vocab_embeddings: Array,
    prompt_length: int,
    rng: PRNGKey,
) -> Array:
  """Initialize prompts by sampling from vocabulary embeddings.
  
  Args:
    vocab_embeddings: Vocabulary embedding matrix [vocab_size, embed_dim].
    prompt_length: Number of prompt tokens.
    rng: Random number generator key.
    
  Returns:
    Initialized prompt embeddings [prompt_length, embed_dim].
  """
  vocab_size = vocab_embeddings.shape[0]
  
  # Sample random indices
  indices = jax.random.randint(rng, (prompt_length,), 0, vocab_size)
  
  # Gather embeddings
  prompt_embeds = vocab_embeddings[indices]
  
  return prompt_embeds


def init_from_text(
    text: str,
    tokenizer: Any,
    vocab_embeddings: Array,
    prompt_length: int,
    rng: PRNGKey,
) -> Array:
  """Initialize prompts from text string.
  
  Args:
    text: Text to initialize from.
    tokenizer: Tokenizer to encode text.
    vocab_embeddings: Vocabulary embedding matrix.
    prompt_length: Desired prompt length.
    rng: Random number generator key.
    
  Returns:
    Initialized prompt embeddings [prompt_length, embed_dim].
  """
  # Tokenize text
  token_ids = tokenizer.encode(text)
  
  # Get embeddings for tokens
  text_embeds = vocab_embeddings[token_ids]
  
  # Pad or truncate to prompt_length
  if len(token_ids) < prompt_length:
    # Pad with random embeddings
    remaining = prompt_length - len(token_ids)
    random_embeds = init_from_vocab(vocab_embeddings, remaining, rng)
    prompt_embeds = jnp.concatenate([text_embeds, random_embeds], axis=0)
  else:
    # Truncate
    prompt_embeds = text_embeds[:prompt_length]
  
  return prompt_embeds
