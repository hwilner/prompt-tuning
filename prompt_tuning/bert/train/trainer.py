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

"""Training utilities for BERT prompt tuning."""

from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm


Array = jax.Array
PRNGKey = jax.random.PRNGKey


class PromptTrainState(train_state.TrainState):
  """Training state for prompt tuning."""
  base_params: Dict[str, Any]
  prompt_params: Dict[str, Any]


def create_train_state(
    model: Any,
    prompt_params: Dict[str, Any],
    base_params: Dict[str, Any],
    learning_rate: float = 0.3,
    weight_decay: float = 0.0,
) -> PromptTrainState:
  """Create training state for prompt tuning."""
  if weight_decay > 0:
    optimizer = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
  else:
    optimizer = optax.adam(learning_rate=learning_rate)
  
  state = PromptTrainState.create(
      apply_fn=model.apply,
      params=prompt_params,
      tx=optimizer,
      base_params=base_params,
      prompt_params=prompt_params,
  )
  
  return state


def compute_classification_loss(
    logits: Array,
    labels: Array,
) -> Tuple[Array, Dict[str, Array]]:
  """Compute cross-entropy loss for classification."""
  # Compute cross-entropy loss
  log_probs = jax.nn.log_softmax(logits, axis=-1)
  
  # One-hot encode labels if needed
  if labels.ndim == 1:
    num_classes = logits.shape[-1]
    labels_one_hot = jax.nn.one_hot(labels, num_classes)
  else:
    labels_one_hot = labels
  
  # Compute loss
  loss = -jnp.sum(labels_one_hot * log_probs, axis=-1)
  loss = jnp.mean(loss)
  
  # Compute accuracy
  predictions = jnp.argmax(logits, axis=-1)
  true_labels = jnp.argmax(labels_one_hot, axis=-1) if labels.ndim > 1 else labels
  accuracy = jnp.mean(predictions == true_labels)
  
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  
  return loss, metrics


@jax.jit
def train_step(
    state: PromptTrainState,
    batch: Dict[str, Array],
) -> Tuple[PromptTrainState, Dict[str, Array]]:
  """Single training step."""
  def loss_fn(prompt_params):
    full_params = {**state.base_params, **prompt_params}
    
    logits = state.apply_fn(
        {'params': full_params},
        batch['input_ids'],
        attention_mask=batch.get('attention_mask'),
        training=True,
    )
    
    loss, metrics = compute_classification_loss(logits, batch['labels'])
    return loss, metrics
  
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, metrics), grads = grad_fn(state.prompt_params)
  
  state = state.apply_gradients(grads=grads)
  
  return state, metrics


@jax.jit
def eval_step(
    state: PromptTrainState,
    batch: Dict[str, Array],
) -> Dict[str, Array]:
  """Single evaluation step."""
  full_params = {**state.base_params, **state.prompt_params}
  
  logits = state.apply_fn(
      {'params': full_params},
      batch['input_ids'],
      attention_mask=batch.get('attention_mask'),
      training=False,
  )
  
  _, metrics = compute_classification_loss(logits, batch['labels'])
  
  return metrics


def train_epoch(
    state: PromptTrainState,
    train_ds: Iterator[Dict[str, Array]],
    num_steps: Optional[int] = None,
) -> Tuple[PromptTrainState, Dict[str, float]]:
  """Train for one epoch."""
  metrics_history = []
  
  iterator = enumerate(train_ds)
  if num_steps is not None:
    iterator = tqdm(iterator, total=num_steps, desc="Training")
  
  for step, batch in iterator:
    if num_steps is not None and step >= num_steps:
      break
    
    state, metrics = train_step(state, batch)
    metrics_history.append(metrics)
    
    if step % 100 == 0:
      avg_loss = jnp.mean(jnp.array([m['loss'] for m in metrics_history[-100:]]))
      print(f"Step {step}, Loss: {avg_loss:.4f}")
  
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
  """Evaluate the model."""
  metrics_history = []
  
  iterator = enumerate(eval_ds)
  if num_steps is not None:
    iterator = tqdm(iterator, total=num_steps, desc="Evaluating")
  
  for step, batch in iterator:
    if num_steps is not None and step >= num_steps:
      break
    
    metrics = eval_step(state, batch)
    metrics_history.append(metrics)
  
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
  """Train prompt tuning end-to-end."""
  state = create_train_state(
      model,
      prompt_params,
      base_params,
      learning_rate,
      weight_decay,
  )
  
  history = {
      'train_loss': [],
      'train_accuracy': [],
      'eval_loss': [],
      'eval_accuracy': [],
  }
  
  for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
    state, train_metrics = train_epoch(state, train_ds, steps_per_epoch)
    history['train_loss'].append(train_metrics['loss'])
    history['train_accuracy'].append(train_metrics['accuracy'])
    
    print(f"Train Loss: {train_metrics['loss']:.4f}, "
          f"Train Accuracy: {train_metrics['accuracy']:.4f}")
    
    if eval_ds is not None and (epoch + 1) % eval_every == 0:
      eval_metrics = evaluate(state, eval_ds)
      history['eval_loss'].append(eval_metrics['loss'])
      history['eval_accuracy'].append(eval_metrics['accuracy'])
      
      print(f"Eval Loss: {eval_metrics['loss']:.4f}, "
            f"Eval Accuracy: {eval_metrics['accuracy']:.4f}")
  
  return state.prompt_params, history


def save_prompt(prompt_params: Dict[str, Any], path: str):
  """Save prompt parameters to file."""
  import pickle
  
  with open(path, 'wb') as f:
    pickle.dump(prompt_params, f)
  
  print(f"Prompt saved to {path}")


def load_prompt(path: str) -> Dict[str, Any]:
  """Load prompt parameters from file."""
  import pickle
  
  with open(path, 'rb') as f:
    prompt_params = pickle.load(f)
  
  print(f"Prompt loaded from {path}")
  return prompt_params
