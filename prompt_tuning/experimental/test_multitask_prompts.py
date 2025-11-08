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

"""Tests for Multi-Task Prompt Tuning."""

import jax
import jax.numpy as jnp
import pytest
from prompt_tuning.experimental import multitask_prompts


class TestMultiTaskPrompt:
  """Tests for MultiTaskPrompt module."""

  def test_concat_composition(self):
    """Test concatenation composition method."""
    module = multitask_prompts.MultiTaskPrompt(
        num_tasks=3,
        shared_length=10,
        task_length=10,
        embed_dim=768,
        composition_method='concat')
    
    rng = jax.random.PRNGKey(0)
    params = module.init(rng, task_id=0)
    
    for task_id in range(3):
      prompt = module.apply(params, task_id=task_id)
      assert prompt.shape == (20, 768)

  def test_add_composition(self):
    """Test addition composition method."""
    module = multitask_prompts.MultiTaskPrompt(
        num_tasks=3,
        shared_length=15,
        task_length=15,
        embed_dim=768,
        composition_method='add')
    
    rng = jax.random.PRNGKey(0)
    params = module.init(rng, task_id=0)
    
    prompt = module.apply(params, task_id=0)
    assert prompt.shape == (15, 768)

  def test_weighted_composition(self):
    """Test weighted composition method."""
    module = multitask_prompts.MultiTaskPrompt(
        num_tasks=3,
        shared_length=15,
        task_length=15,
        embed_dim=768,
        composition_method='weighted')
    
    rng = jax.random.PRNGKey(0)
    params = module.init(rng, task_id=0)
    
    prompt = module.apply(params, task_id=0)
    assert prompt.shape == (15, 768)

  def test_different_tasks_different_prompts(self):
    """Test that different tasks get different prompts."""
    module = multitask_prompts.MultiTaskPrompt(
        num_tasks=3,
        shared_length=10,
        task_length=10,
        embed_dim=768,
        composition_method='concat')
    
    rng = jax.random.PRNGKey(0)
    params = module.init(rng, task_id=0)
    
    prompt_0 = module.apply(params, task_id=0)
    prompt_1 = module.apply(params, task_id=1)
    
    # Prompts should be different due to task-specific components
    assert not jnp.allclose(prompt_0, prompt_1)


class TestHierarchicalMultiTaskPrompt:
  """Tests for HierarchicalMultiTaskPrompt module."""

  def test_hierarchical_generation(self):
    """Test hierarchical prompt generation."""
    module = multitask_prompts.HierarchicalMultiTaskPrompt(
        num_groups=3,
        tasks_per_group=[2, 3, 2],
        global_length=5,
        group_length=5,
        task_length=5,
        embed_dim=768)
    
    rng = jax.random.PRNGKey(0)
    params = module.init(rng, group_id=0, task_id=0)
    
    prompt = module.apply(params, group_id=0, task_id=0)
    assert prompt.shape == (15, 768)

  def test_different_groups_different_prompts(self):
    """Test that different groups get different prompts."""
    module = multitask_prompts.HierarchicalMultiTaskPrompt(
        num_groups=3,
        tasks_per_group=[2, 3, 2],
        global_length=5,
        group_length=5,
        task_length=5,
        embed_dim=768)
    
    rng = jax.random.PRNGKey(0)
    params = module.init(rng, group_id=0, task_id=0)
    
    prompt_g0 = module.apply(params, group_id=0, task_id=0)
    prompt_g1 = module.apply(params, group_id=1, task_id=0)
    
    assert not jnp.allclose(prompt_g0, prompt_g1)


class TestAdaptiveMultiTaskPrompt:
  """Tests for AdaptiveMultiTaskPrompt module."""

  def test_adaptive_generation(self):
    """Test adaptive prompt generation."""
    module = multitask_prompts.AdaptiveMultiTaskPrompt(
        num_tasks=5,
        num_components=20,
        component_length=10,
        embed_dim=768,
        num_heads=4)
    
    rng = jax.random.PRNGKey(0)
    params = module.init(rng, task_id=0)
    
    prompt = module.apply(params, task_id=0)
    assert prompt.shape == (10, 768)


class TestUtilities:
  """Tests for utility functions."""

  def test_get_total_length_concat(self):
    """Test total length calculation for concat method."""
    length = multitask_prompts.get_multitask_prompt_total_length(
        shared_length=10,
        task_length=10,
        composition_method='concat')
    assert length == 20

  def test_get_total_length_add(self):
    """Test total length calculation for add method."""
    length = multitask_prompts.get_multitask_prompt_total_length(
        shared_length=15,
        task_length=15,
        composition_method='add')
    assert length == 15


if __name__ == '__main__':
  pytest.main([__file__])
