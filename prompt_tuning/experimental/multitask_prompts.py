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

"""Multi-Task Prompt Tuning with shared and task-specific components.

This module implements a multi-task prompt tuning approach where prompts are
composed of shared components (common across tasks) and task-specific components.
This enables knowledge transfer between related tasks while maintaining
task-specific adaptations.

Reference:
  Wang et al. (2022). "Multitask Prompt Tuning Enables Parameter-Efficient
  Transfer Learning."
  https://arxiv.org/abs/2303.02861
"""

from typing import Callable, Dict, Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp
from flaxformer.types import Array, DType


class MultiTaskPrompt(nn.Module):
  """Multi-task prompt with shared and task-specific components.
  
  The prompt is composed of:
  - A shared component learned across all tasks
  - Task-specific components for each individual task
  - Optional task embeddings for dynamic composition
  
  Attributes:
    num_tasks: Number of tasks.
    shared_length: Length of the shared prompt component.
    task_length: Length of each task-specific prompt component.
    embed_dim: Embedding dimension.
    prompt_init: Initializer for prompt embeddings.
    dtype: Data type for parameters.
    use_task_embeddings: Whether to use learnable task embeddings.
    composition_method: How to combine shared and task-specific prompts
      ('concat', 'add', 'weighted', 'gated').
  """
  num_tasks: int
  shared_length: int
  task_length: int
  embed_dim: int
  prompt_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.uniform()
  dtype: DType = jnp.float32
  use_task_embeddings: bool = False
  composition_method: str = 'concat'

  @nn.compact
  def __call__(self, task_id: int) -> Array:
    """Generate prompt for a specific task.
    
    Args:
      task_id: Integer ID of the task (0 to num_tasks-1).
      
    Returns:
      Prompt for the specified task with shape [prompt_length, embed_dim].
      The prompt_length depends on the composition method.
    """
    # Shared prompt component
    shared_prompt = self.param(
        'shared_prompt',
        self.prompt_init,
        (self.shared_length, self.embed_dim),
        self.dtype)
    
    # Task-specific prompts
    task_prompts = self.param(
        'task_prompts',
        self.prompt_init,
        (self.num_tasks, self.task_length, self.embed_dim),
        self.dtype)
    
    # Get the task-specific component
    task_prompt = task_prompts[task_id]  # [task_length, embed_dim]
    
    # Combine shared and task-specific components
    if self.composition_method == 'concat':
      # Simple concatenation
      return jnp.concatenate([shared_prompt, task_prompt], axis=0)
    
    elif self.composition_method == 'add':
      # Element-wise addition (requires same length)
      if self.shared_length != self.task_length:
        raise ValueError(
            f"For 'add' composition, shared_length ({self.shared_length}) "
            f"must equal task_length ({self.task_length})")
      return shared_prompt + task_prompt
    
    elif self.composition_method == 'weighted':
      # Weighted combination with learnable weights
      if self.shared_length != self.task_length:
        raise ValueError(
            f"For 'weighted' composition, shared_length ({self.shared_length}) "
            f"must equal task_length ({self.task_length})")
      
      # Learnable mixing weight per task
      mixing_weights = self.param(
          'mixing_weights',
          nn.initializers.constant(0.5),
          (self.num_tasks,),
          self.dtype)
      
      weight = jnp.sigmoid(mixing_weights[task_id])
      return weight * shared_prompt + (1 - weight) * task_prompt
    
    elif self.composition_method == 'gated':
      # Gated combination using a small MLP
      if self.shared_length != self.task_length:
        raise ValueError(
            f"For 'gated' composition, shared_length ({self.shared_length}) "
            f"must equal task_length ({self.task_length})")
      
      # Concatenate and pass through gate network
      combined = jnp.stack([shared_prompt, task_prompt], axis=0)  # [2, L, H]
      combined_flat = jnp.reshape(combined, (-1,))  # [2*L*H]
      
      gate_hidden = nn.Dense(
          features=self.embed_dim,
          dtype=self.dtype,
          name=f'gate_hidden_{task_id}')(combined_flat)
      gate_hidden = nn.gelu(gate_hidden)
      
      gate_logits = nn.Dense(
          features=2,
          dtype=self.dtype,
          name=f'gate_output_{task_id}')(gate_hidden)
      gate_weights = nn.softmax(gate_logits)  # [2]
      
      # Apply gate weights
      return (gate_weights[0] * shared_prompt + 
              gate_weights[1] * task_prompt)
    
    else:
      raise ValueError(
          f"Unknown composition_method: {self.composition_method}. "
          f"Must be one of: 'concat', 'add', 'weighted', 'gated'")


class HierarchicalMultiTaskPrompt(nn.Module):
  """Hierarchical multi-task prompts with task groups.
  
  This variant organizes tasks into groups (e.g., by domain or task type)
  and uses a three-level hierarchy:
  - Global shared component (all tasks)
  - Group-specific component (tasks in the same group)
  - Task-specific component (individual task)
  
  Attributes:
    num_groups: Number of task groups.
    tasks_per_group: Number of tasks in each group.
    global_length: Length of global shared prompt.
    group_length: Length of group-specific prompts.
    task_length: Length of task-specific prompts.
    embed_dim: Embedding dimension.
    prompt_init: Initializer for prompts.
    dtype: Data type for parameters.
  """
  num_groups: int
  tasks_per_group: Sequence[int]
  global_length: int
  group_length: int
  task_length: int
  embed_dim: int
  prompt_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.uniform()
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, group_id: int, task_id: int) -> Array:
    """Generate prompt for a specific task in a group.
    
    Args:
      group_id: Integer ID of the task group.
      task_id: Integer ID of the task within the group.
      
    Returns:
      Hierarchical prompt with shape [global_length + group_length + task_length, embed_dim].
    """
    # Global shared prompt
    global_prompt = self.param(
        'global_prompt',
        self.prompt_init,
        (self.global_length, self.embed_dim),
        self.dtype)
    
    # Group-specific prompts
    group_prompts = self.param(
        'group_prompts',
        self.prompt_init,
        (self.num_groups, self.group_length, self.embed_dim),
        self.dtype)
    
    # Task-specific prompts (flattened across all groups)
    total_tasks = sum(self.tasks_per_group)
    task_prompts = self.param(
        'task_prompts',
        self.prompt_init,
        (total_tasks, self.task_length, self.embed_dim),
        self.dtype)
    
    # Get group-specific component
    group_prompt = group_prompts[group_id]
    
    # Calculate global task ID
    global_task_id = sum(self.tasks_per_group[:group_id]) + task_id
    task_prompt = task_prompts[global_task_id]
    
    # Concatenate all components
    return jnp.concatenate(
        [global_prompt, group_prompt, task_prompt],
        axis=0)


class AdaptiveMultiTaskPrompt(nn.Module):
  """Adaptive multi-task prompts with attention-based composition.
  
  This variant uses attention mechanisms to dynamically compose prompts
  from a pool of prompt components based on task characteristics.
  
  Attributes:
    num_tasks: Number of tasks.
    num_components: Number of prompt components in the pool.
    component_length: Length of each prompt component.
    embed_dim: Embedding dimension.
    num_heads: Number of attention heads.
    prompt_init: Initializer for prompts.
    dtype: Data type for parameters.
  """
  num_tasks: int
  num_components: int
  component_length: int
  embed_dim: int
  num_heads: int = 4
  prompt_init: Callable[[Array, Sequence[int]], Array] = nn.initializers.uniform()
  dtype: DType = jnp.float32

  @nn.compact
  def __call__(self, task_id: int) -> Array:
    """Generate adaptive prompt for a specific task.
    
    Args:
      task_id: Integer ID of the task.
      
    Returns:
      Adaptively composed prompt with shape [component_length, embed_dim].
    """
    # Pool of prompt components
    prompt_components = self.param(
        'prompt_components',
        self.prompt_init,
        (self.num_components, self.component_length, self.embed_dim),
        self.dtype)
    
    # Task embeddings (queries for attention)
    task_embeddings = self.param(
        'task_embeddings',
        nn.initializers.normal(stddev=0.02),
        (self.num_tasks, self.embed_dim),
        self.dtype)
    
    # Get task embedding
    task_emb = task_embeddings[task_id]  # [embed_dim]
    
    # Compute attention weights over prompt components
    # Use multi-head attention for richer composition
    query = jnp.expand_dims(task_emb, axis=0)  # [1, embed_dim]
    
    # Flatten components for attention
    keys = jnp.reshape(
        prompt_components,
        (self.num_components * self.component_length, self.embed_dim))
    values = keys
    
    # Simple scaled dot-product attention
    attention_logits = jnp.matmul(query, keys.T)  # [1, num_components * component_length]
    attention_logits = attention_logits / jnp.sqrt(self.embed_dim)
    attention_weights = nn.softmax(attention_logits, axis=-1)
    
    # Weighted combination of components
    attended_prompt = jnp.matmul(attention_weights, values)  # [1, embed_dim]
    attended_prompt = jnp.squeeze(attended_prompt, axis=0)  # [embed_dim]
    
    # Expand to full prompt length
    prompt = jnp.tile(
        jnp.expand_dims(attended_prompt, axis=0),
        [self.component_length, 1])
    
    # Add residual connection with a base component
    base_component = prompt_components[0]  # Use first component as base
    return prompt + base_component


def get_multitask_prompt_total_length(
    shared_length: int,
    task_length: int,
    composition_method: str) -> int:
  """Calculate total prompt length for multi-task prompts.
  
  Args:
    shared_length: Length of shared component.
    task_length: Length of task-specific component.
    composition_method: Composition method used.
    
  Returns:
    Total length of the composed prompt.
  """
  if composition_method == 'concat':
    return shared_length + task_length
  elif composition_method in ['add', 'weighted', 'gated']:
    return shared_length  # Assumes shared_length == task_length
  else:
    raise ValueError(f"Unknown composition_method: {composition_method}")
