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

"""Training utilities for Gemma prompt tuning.

This module provides training loops, optimization, and evaluation utilities
for prompt tuning with Gemma models.
"""

from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm


Array = jax.Array
PRNGKey = jax.random.PRNGKey


class PromptTrainState(train_state.TrainState):
  """Training state for prompt tuning.
  
  Extends Flax's TrainState with prompt-specific functionality.
  
  Attributes:
    base_params: Frozen base model parameters.
    prompt_params: Trainable prompt parameters.
  """
  base_params: Dict[str, Any]
  prompt_params: Dict[str, Any]


def create_train_state(
    model: Any,
    prompt_params: Dict[str, Any],
    base_params: Dict[str, Any],
    learning_rate: float = 0.3,
    weight_decay: float = 0.0,
) -> PromptTrainState:
  """Create training state for prompt tuning.
  
  Args:
    model: The prompt-tuned Gemma model.
    prompt_params: Initial prompt parameters.
    base_params: Frozen base model parameters.
    learning_rate: Learning rate for prompt optimization.
    weight_decay: Weight decay for regularization.
    
  Returns:
    Initialized training state.
  """
  # Create optimizer for prompts only
  if weight_decay > 0:
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
  else:
    optimizer = optax.adam(learning_rate=learning_rate)
  
  # Create training state
  state = PromptTrainState.create(
      apply_fn=model.apply,
      params=prompt_params,
      tx=optimizer,
      base_params=base_params,
      prompt_params=prompt_params,
  )
  
  return state


def compute_loss(
    logits: Array,
    labels: Array,
    mask: Optional[Array] = None,
) -> Tuple[Array, Dict[str, Array]]:
  """Compute cross-entropy loss for language modeling.
  
  Args:
    logits: Model logits of shape [batch_size, seq_len, vocab_size].
    labels: Target labels of shape [batch_size, seq_len].
    mask: Optional mask for valid positions.
    
  Returns:
    Tuple of (loss, metrics_dict).
  """
  # Compute cross-entropy loss
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  
  # Gather log probabilities for target labels
  batch_size, seq_len = labels.shape
  indices = jnp.arange(batch_size * seq_len)
  flat_labels = labels.reshape(-1)
  flat_log_probs = log_probs.reshape(-1, logits.shape[-1])
  
  target_log_probs = flat_log_probs[indices, flat_labels]
  target_log_probs = target_log_probs.reshape(batch_size, seq_len)
  
  # Apply mask if provided
  if mask is not None:
    target_log_probs = target_log_probs * mask
    num_tokens = jnp.sum(mask)
  else:
    num_tokens = batch_size * seq_len
  
  # Compute loss
  loss = -jnp.sum(target_log_probs) / num_tokens
  
  # Compute accuracy
  predictions = jnp.argmax(logits, axis=-1)
  correct = (predictions == labels).astype(jnp.float32)
  if mask is not None:
    correct = correct * mask
  accuracy = jnp.sum(correct) / num_tokens
  
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'perplexity': jnp.exp(loss),
  }
  
  return loss, metrics


@jax.jit
def train_step(
    state: PromptTrainState,
    batch: Dict[str, Array],
) -> Tuple[PromptTrainState, Dict[str, Array]]:
  """Single training step.
  
  Args:
    state: Current training state.
    batch: Batch of data with 'input_ids' and 'labels'.
    
  Returns:
    Tuple of (updated_state, metrics).
  """
  def loss_fn(prompt_params):
    # Combine prompt and base parameters
    full_params = {**state.base_params, **prompt_params}
    
    # Forward pass
    logits = state.apply_fn(
        {'params': full_params},
        batch['input_ids'],
    )
    
    # Compute loss
    loss, metrics = compute_loss(
        logits,
        batch['labels'],
        batch.get('mask'),
    )
    
    return loss, metrics
  
  # Compute gradients
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, metrics), grads = grad_fn(state.prompt_params)
  
  # Update parameters
  state = state.apply_gradients(grads=grads)
  
  return state, metrics


@jax.jit
def eval_step(
    state: PromptTrainState,
    batch: Dict[str, Array],
) -> Dict[str, Array]:
  """Single evaluation step.
  
  Args:
    state: Current training state.
    batch: Batch of data with 'input_ids' and 'labels'.
    
  Returns:
    Evaluation metrics.
  """
  # Combine prompt and base parameters
  full_params = {**state.base_params, **state.prompt_params}
  
  # Forward pass
  logits = state.apply_fn(
      {'params': full_params},
      batch['input_ids'],
  )
  
  # Compute loss
  _, metrics = compute_loss(
      logits,
      batch['labels'],
      batch.get('mask'),
  )
  
  return metrics


def train_epoch(
    state: PromptTrainState,
    train_ds: Iterator[Dict[str, Array]],
    num_steps: Optional[int] = None,
) -> Tuple[PromptTrainState, Dict[str, float]]:
  """Train for one epoch.
  
  Args:
    state: Current training state.
    train_ds: Training dataset iterator.
    num_steps: Number of steps to train. If None, train until dataset exhausted.
    
  Returns:
    Tuple of (updated_state, average_metrics).
  """
  metrics_history = []
  
  iterator = enumerate(train_ds)
  if num_steps is not None:
    iterator = tqdm(iterator, total=num_steps, desc="Training")
  
  for step, batch in iterator:
    if num_steps is not None and step >= num_steps:
      break
    
    state, metrics = train_step(state, batch)
    metrics_history.append(metrics)
    
    # Log progress
    if step % 100 == 0:
      avg_loss = jnp.mean(jnp.array([m['loss'] for m in metrics_history[-100:]]))
      print(f"Step {step}, Loss: {avg_loss:.4f}")
  
  # Compute average metrics
  avg_metrics = {
      key: float(jnp.mean(jnp.array([m[key] for m in metrics_history])))
      for key in metrics_history[0].keys()
  }
  
  return state, avg_metrics


def evaluate(
    state: PromptTrainState,
    eval_ds: Iterator[Dict[str, Array]],
    num_steps: Optional[int] = None,
) -> Dict[str, float]:
  """Evaluate the model.
  
  Args:
    state: Current training state.
    eval_ds: Evaluation dataset iterator.
    num_steps: Number of steps to evaluate. If None, evaluate entire dataset.
    
  Returns:
    Average evaluation metrics.
  """
  metrics_history = []
  
  iterator = enumerate(eval_ds)
  if num_steps is not None:
    iterator = tqdm(iterator, total=num_steps, desc="Evaluating")
  
  for step, batch in iterator:
    if num_steps is not None and step >= num_steps:
      break
    
    metrics = eval_step(state, batch)
    metrics_history.append(metrics)
  
  # Compute average metrics
  avg_metrics = {
      key: float(jnp.mean(jnp.array([m[key] for m in metrics_history])))
      for key in metrics_history[0].keys()
  }
  
  return avg_metrics


def train_prompt(
    model: Any,
    prompt_params: Dict[str, Any],
    base_params: Dict[str, Any],
    train_ds: Iterator[Dict[str, Array]],
    eval_ds: Optional[Iterator[Dict[str, Array]]] = None,
    num_epochs: int = 3,
    steps_per_epoch: Optional[int] = None,
    learning_rate: float = 0.3,
    weight_decay: float = 0.0,
    eval_every: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """Train prompt tuning end-to-end.
  
  Args:
    model: The prompt-tuned Gemma model.
    prompt_params: Initial prompt parameters.
    base_params: Frozen base model parameters.
    train_ds: Training dataset iterator.
    eval_ds: Optional evaluation dataset iterator.
    num_epochs: Number of training epochs.
    steps_per_epoch: Steps per epoch. If None, use entire dataset.
    learning_rate: Learning rate for optimization.
    weight_decay: Weight decay for regularization.
    eval_every: Evaluate every N epochs.
    
  Returns:
    Tuple of (trained_prompt_params, training_history).
  """
  # Create training state
  state = create_train_state(
      model,
      prompt_params,
      base_params,
      learning_rate,
      weight_decay,
  )
  
  # Training history
  history = {
      'train_loss': [],
      'train_accuracy': [],
      'eval_loss': [],
      'eval_accuracy': [],
  }
  
  # Training loop
  for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
    # Train
    state, train_metrics = train_epoch(state, train_ds, steps_per_epoch)
    history['train_loss'].append(train_metrics['loss'])
    history['train_accuracy'].append(train_metrics['accuracy'])
    
    print(f"Train Loss: {train_metrics['loss']:.4f}, "
          f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    
    # Evaluate
    if eval_ds is not None and (epoch + 1) % eval_every == 0:
      eval_metrics = evaluate(state, eval_ds)
      history['eval_loss'].append(eval_metrics['loss'])
      history['eval_accuracy'].append(eval_metrics['accuracy'])
      
      print(f"Eval Loss: {eval_metrics['loss']:.4f}, "
            f"Eval Accuracy: {eval_metrics['accuracy']:.4f}")
  
  return state.prompt_params, history


def save_prompt(prompt_params: Dict[str, Any], path: str):
  """Save prompt parameters to file.
  
  Args:
    prompt_params: Prompt parameters to save.
    path: Path to save to.
  """
  import pickle
  
  with open(path, 'wb') as f:
    pickle.dump(prompt_params, f)
  
  print(f"Prompt saved to {path}")


def load_prompt(path: str) -> Dict[str, Any]:
  """Load prompt parameters from file.
  
  Args:
    path: Path to load from.
    
  Returns:
    Loaded prompt parameters.
  """
  import pickle
  
  with open(path, 'rb') as f:
    prompt_params = pickle.load(f)
  
  print(f"Prompt loaded from {path}")
  return prompt_params
