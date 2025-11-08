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

"""Tests for P-Tuning v2 (Deep Prompt Tuning)."""

import jax
import jax.numpy as jnp
import pytest
from prompt_tuning.experimental import p_tuning_v2


class TestDeepPrompt:
  """Tests for DeepPrompt module."""

  def test_basic_deep_prompt(self):
    """Test basic deep prompt generation."""
    module = p_tuning_v2.DeepPrompt(
        num_layers=12,
        prompt_length=20,
        embed_dim=768,
        shared_prompts=False)
    
    rng = jax.random.PRNGKey(0)
    params = module.init(rng)
    prompts = module.apply(params)
    
    assert prompts.shape == (12, 20, 768)

  def test_shared_deep_prompt(self):
    """Test deep prompt with shared prompts across layers."""
    module = p_tuning_v2.DeepPrompt(
        num_layers=12,
        prompt_length=20,
        embed_dim=768,
        shared_prompts=True)
    
    rng = jax.random.PRNGKey(0)
    params = module.init(rng)
    prompts = module.apply(params)
    
    assert prompts.shape == (12, 20, 768)
    # Check that all layers have the same prompt
    for i in range(1, 12):
      assert jnp.allclose(prompts[0], prompts[i])


class TestDeepPromptEncoder:
  """Tests for DeepPromptEncoder module."""

  def test_encoder_generation(self):
    """Test MLP-based prompt encoder."""
    module = p_tuning_v2.DeepPromptEncoder(
        num_layers=12,
        prompt_length=20,
        embed_dim=768,
        hidden_dim=512)
    
    rng = jax.random.PRNGKey(0)
    params = module.init(rng)
    prompts = module.apply(params)
    
    assert prompts.shape == (12, 20, 768)

  def test_encoder_different_layers(self):
    """Test that encoder generates different prompts for different layers."""
    module = p_tuning_v2.DeepPromptEncoder(
        num_layers=12,
        prompt_length=20,
        embed_dim=768,
        hidden_dim=512)
    
    rng = jax.random.PRNGKey(0)
    params = module.init(rng)
    prompts = module.apply(params)
    
    # Check that different layers have different prompts
    assert not jnp.allclose(prompts[0], prompts[1])


class TestDeepPromptUtilities:
  """Tests for utility functions."""

  def test_expand_to_batch(self):
    """Test expanding prompts to batch size."""
    deep_prompts = jnp.ones((12, 20, 768))
    batch_size = 8
    
    batched = p_tuning_v2.expand_deep_prompts_to_batch(deep_prompts, batch_size)
    
    assert batched.shape == (12, 8, 20, 768)

  def test_prefix_deep_prompt(self):
    """Test prefixing layer input with deep prompt."""
    batch_size = 8
    seq_length = 128
    embed_dim = 768
    prompt_length = 20
    
    deep_prompts = jnp.ones((12, batch_size, prompt_length, embed_dim))
    layer_input = jnp.zeros((batch_size, seq_length, embed_dim))
    
    result = p_tuning_v2.prefix_deep_prompt(deep_prompts, layer_idx=5, x_embed=layer_input)
    
    assert result.shape == (batch_size, prompt_length + seq_length, embed_dim)


class TestDeepPromptConfig:
  """Tests for DeepPromptConfig."""

  def test_config_creation(self):
    """Test creating configuration."""
    config = p_tuning_v2.DeepPromptConfig(
        num_layers=12,
        prompt_length=20,
        embed_dim=768,
        shared_prompts=False,
        use_encoder=False)
    
    assert config.num_layers == 12
    assert config.prompt_length == 20
    assert config.embed_dim == 768
    assert not config.shared_prompts
    assert not config.use_encoder

  def test_create_module_basic(self):
    """Test creating module from config."""
    config = p_tuning_v2.DeepPromptConfig(
        num_layers=12,
        prompt_length=20,
        embed_dim=768,
        use_encoder=False)
    
    module = p_tuning_v2.create_deep_prompt_module(config)
    assert isinstance(module, p_tuning_v2.DeepPrompt)

  def test_create_module_encoder(self):
    """Test creating encoder module from config."""
    config = p_tuning_v2.DeepPromptConfig(
        num_layers=12,
        prompt_length=20,
        embed_dim=768,
        use_encoder=True,
        encoder_hidden_dim=512)
    
    module = p_tuning_v2.create_deep_prompt_module(config)
    assert isinstance(module, p_tuning_v2.DeepPromptEncoder)


if __name__ == '__main__':
  pytest.main([__file__])
