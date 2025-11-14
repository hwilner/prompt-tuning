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

"""Prompt tuning implementation for BERT models.

This module provides soft prompt tuning capabilities for BERT encoder-only
transformer models. BERT uses bidirectional attention, making it suitable for
tasks like classification, named entity recognition, and question answering.

Key features:
  - Soft prompts prepended to input embeddings
  - Compatible with BERT's learned position embeddings
  - Supports bidirectional attention
  - Works with BERT-base and BERT-large

Example usage:
  ```python
  from transformers import FlaxBertModel
  from prompt_tuning.bert import prompts
  
  # Load base BERT model
  model = FlaxBertModel.from_pretrained('bert-base-uncased')
  
  # Create prompt-tuned model
  prompt_config = prompts.PromptConfig(
      prompt_length=20,
      embed_dim=768,  # BERT-base
  )
  prompt_model = prompts.PromptBERT(model, prompt_config)
  
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
  """Configuration for prompt tuning with BERT.
  
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
  """Learnable soft prompt module for BERT.
  
  This module creates trainable prompt embeddings that are prepended to the
  input token embeddings. The prompts are learned during training while the
  base BERT model parameters remain frozen.
  
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


class PromptBERT(nn.Module):
  """BERT model with prompt tuning.
  
  This wrapper adds soft prompt tuning capabilities to a BERT model. The
  prompts are prepended to the input embeddings, and position indices and
  attention masks are adjusted accordingly.
  
  Attributes:
    base_model: The underlying BERT model.
    prompt_config: Configuration for the prompt.
  """
  
  base_model: Any  # BERT model from transformers
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
      attention_mask: Optional[Array] = None,
      token_type_ids: Optional[Array] = None,
      position_ids: Optional[Array] = None,
      return_dict: bool = True,
      **kwargs,
  ) -> Any:
    """Forward pass with prompt tuning.
    
    Args:
      input_ids: Input token IDs of shape [batch_size, seq_len].
      attention_mask: Attention mask of shape [batch_size, seq_len].
      token_type_ids: Token type IDs for segment embeddings.
      position_ids: Position IDs for position embeddings.
      return_dict: Whether to return a dictionary or tuple.
      **kwargs: Additional arguments passed to the base model.
      
    Returns:
      Model output (same format as base BERT model).
    """
    batch_size, seq_len = input_ids.shape
    prompt_length = self.prompt_config.prompt_length
    
    # Get input embeddings from base model
    # Note: This assumes the base model has an 'embeddings' module
    # We'll get word embeddings and add position/token type embeddings later
    input_embeds = self.base_model.embeddings.word_embeddings(input_ids)
    
    # Generate soft prompts
    prompt_embeds = self.prompt(batch_size)
    
    # Concatenate prompts with input embeddings
    # Shape: [batch_size, prompt_length + seq_len, embed_dim]
    combined_embeds = jnp.concatenate([prompt_embeds, input_embeds], axis=1)
    
    # Adjust attention mask
    if attention_mask is None:
      # Create mask with all ones (attend to everything)
      attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    # Extend attention mask to include prompts
    prompt_mask = jnp.ones((batch_size, prompt_length), dtype=jnp.int32)
    extended_attention_mask = jnp.concatenate(
        [prompt_mask, attention_mask], axis=1
    )
    
    # Adjust token type IDs
    if token_type_ids is not None:
      # Extend with zeros for prompts
      prompt_token_types = jnp.zeros((batch_size, prompt_length), dtype=jnp.int32)
      extended_token_type_ids = jnp.concatenate(
          [prompt_token_types, token_type_ids], axis=1
      )
    else:
      extended_token_type_ids = None
    
    # Adjust position IDs
    if position_ids is None:
      # Create position IDs: [0, 1, ..., prompt_length + seq_len - 1]
      position_ids = jnp.arange(prompt_length + seq_len)
      position_ids = jnp.expand_dims(position_ids, axis=0)
      position_ids = jnp.tile(position_ids, (batch_size, 1))
    else:
      # Shift provided positions by prompt_length
      prompt_positions = jnp.arange(prompt_length)
      prompt_positions = jnp.expand_dims(prompt_positions, axis=0)
      prompt_positions = jnp.tile(prompt_positions, (batch_size, 1))
      position_ids = position_ids + prompt_length
      position_ids = jnp.concatenate([prompt_positions, position_ids], axis=1)
    
    # Add position and token type embeddings to combined embeddings
    position_embeds = self.base_model.embeddings.position_embeddings(position_ids)
    combined_embeds = combined_embeds + position_embeds
    
    if extended_token_type_ids is not None:
      token_type_embeds = self.base_model.embeddings.token_type_embeddings(
          extended_token_type_ids
      )
      combined_embeds = combined_embeds + token_type_embeds
    
    # Apply LayerNorm and dropout
    combined_embeds = self.base_model.embeddings.LayerNorm(combined_embeds)
    combined_embeds = self.base_model.embeddings.dropout(combined_embeds)
    
    # Forward pass through encoder with combined embeddings
    encoder_outputs = self.base_model.encoder(
        combined_embeds,
        attention_mask=extended_attention_mask,
        **kwargs,
    )
    
    # Pool the output (typically use [CLS] token, which is now at position prompt_length)
    sequence_output = encoder_outputs[0]
    
    # For compatibility, we might want to return only the non-prompt part
    # or keep the full sequence including prompts
    # Here we keep the full sequence
    
    if return_dict:
      return {
          'last_hidden_state': sequence_output,
          'hidden_states': encoder_outputs.get('hidden_states'),
          'attentions': encoder_outputs.get('attentions'),
      }
    else:
      return (sequence_output,) + encoder_outputs[1:]
  
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
) -> PromptBERT:
  """Create a prompt-tuned BERT model.
  
  Args:
    base_model: Base BERT model from transformers.
    prompt_config: Configuration for the prompt.
    
  Returns:
    PromptBERT model ready for training or inference.
  """
  return PromptBERT(base_model=base_model, prompt_config=prompt_config)


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
  token_ids = tokenizer.encode(text, add_special_tokens=False)
  
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


# Classification head for prompt-tuned BERT

class PromptBERTForSequenceClassification(nn.Module):
  """BERT with prompt tuning for sequence classification.
  
  This adds a classification head on top of the prompt-tuned BERT model.
  
  Attributes:
    base_model: The prompt-tuned BERT model.
    num_labels: Number of classification labels.
    dropout_rate: Dropout rate for the classifier.
  """
  
  base_model: PromptBERT
  num_labels: int
  dropout_rate: float = 0.1
  
  @nn.compact
  def __call__(
      self,
      input_ids: Array,
      attention_mask: Optional[Array] = None,
      training: bool = False,
      **kwargs,
  ) -> Array:
    """Forward pass with classification head.
    
    Args:
      input_ids: Input token IDs.
      attention_mask: Attention mask.
      training: Whether in training mode.
      **kwargs: Additional arguments.
      
    Returns:
      Logits of shape [batch_size, num_labels].
    """
    # Get BERT outputs
    outputs = self.base_model(
        input_ids,
        attention_mask=attention_mask,
        return_dict=True,
        **kwargs,
    )
    
    # Use [CLS] token representation (at position prompt_length)
    prompt_length = self.base_model.prompt_config.prompt_length
    cls_output = outputs['last_hidden_state'][:, prompt_length, :]
    
    # Apply dropout
    cls_output = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(
        cls_output
    )
    
    # Classification head
    logits = nn.Dense(self.num_labels)(cls_output)
    
    return logits
