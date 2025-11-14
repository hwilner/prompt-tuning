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

"""Tests for Gemma prompt tuning modules."""

import jax
import jax.numpy as jnp
import pytest

from prompt_tuning.gemma import prompts, configs


class TestPromptConfig:
  """Tests for PromptConfig."""
  
  def test_basic_config(self):
    """Test basic configuration creation."""
    config = prompts.PromptConfig(
        prompt_length=20,
        embed_dim=2048,
    )
    
    assert config.prompt_length == 20
    assert config.embed_dim == 2048
    assert config.init_scale == 0.5
    assert config.init_from_vocab == False
    assert config.init_text is None
  
  def test_custom_config(self):
    """Test configuration with custom values."""
    config = prompts.PromptConfig(
        prompt_length=50,
        embed_dim=1024,
        init_scale=0.1,
        init_from_vocab=True,
        init_text="Test prompt",
    )
    
    assert config.prompt_length == 50
    assert config.embed_dim == 1024
    assert config.init_scale == 0.1
    assert config.init_from_vocab == True
    assert config.init_text == "Test prompt"


class TestSoftPrompt:
  """Tests for SoftPrompt module."""
  
  def test_prompt_shape(self):
    """Test that prompt has correct shape."""
    prompt_module = prompts.SoftPrompt(
        prompt_length=20,
        embed_dim=128,
    )
    
    # Initialize
    rng = jax.random.PRNGKey(0)
    batch_size = 4
    
    variables = prompt_module.init(rng, batch_size)
    prompt_embeds = prompt_module.apply(variables, batch_size)
    
    # Check shape
    assert prompt_embeds.shape == (batch_size, 20, 128)
  
  def test_prompt_initialization(self):
    """Test that prompts are initialized with correct scale."""
    prompt_module = prompts.SoftPrompt(
        prompt_length=10,
        embed_dim=64,
        init_scale=0.5,
    )
    
    rng = jax.random.PRNGKey(42)
    variables = prompt_module.init(rng, 1)
    
    # Check that values are within reasonable range
    prompt_params = variables['params']['prompt']
    assert jnp.std(prompt_params) < 1.0  # Should be relatively small


class TestInitializationUtilities:
  """Tests for initialization utilities."""
  
  def test_init_from_vocab(self):
    """Test initialization from vocabulary embeddings."""
    vocab_size = 1000
    embed_dim = 128
    prompt_length = 20
    
    # Create dummy vocabulary embeddings
    vocab_embeddings = jax.random.normal(
        jax.random.PRNGKey(0),
        (vocab_size, embed_dim),
    )
    
    # Initialize prompt
    rng = jax.random.PRNGKey(1)
    prompt_embeds = prompts.init_from_vocab(
        vocab_embeddings,
        prompt_length,
        rng,
    )
    
    # Check shape
    assert prompt_embeds.shape == (prompt_length, embed_dim)
  
  def test_init_from_text_truncate(self):
    """Test initialization from text with truncation."""
    vocab_size = 1000
    embed_dim = 128
    prompt_length = 5
    
    # Create dummy vocabulary embeddings
    vocab_embeddings = jax.random.normal(
        jax.random.PRNGKey(0),
        (vocab_size, embed_dim),
    )
    
    # Mock tokenizer
    class MockTokenizer:
      def encode(self, text):
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens
    
    tokenizer = MockTokenizer()
    
    # Initialize prompt (should truncate to 5 tokens)
    rng = jax.random.PRNGKey(1)
    prompt_embeds = prompts.init_from_text(
        "This is a long text",
        tokenizer,
        vocab_embeddings,
        prompt_length,
        rng,
    )
    
    # Check shape
    assert prompt_embeds.shape == (prompt_length, embed_dim)
  
  def test_init_from_text_pad(self):
    """Test initialization from text with padding."""
    vocab_size = 1000
    embed_dim = 128
    prompt_length = 20
    
    # Create dummy vocabulary embeddings
    vocab_embeddings = jax.random.normal(
        jax.random.PRNGKey(0),
        (vocab_size, embed_dim),
    )
    
    # Mock tokenizer
    class MockTokenizer:
      def encode(self, text):
        return [1, 2, 3]  # 3 tokens
    
    tokenizer = MockTokenizer()
    
    # Initialize prompt (should pad to 20 tokens)
    rng = jax.random.PRNGKey(1)
    prompt_embeds = prompts.init_from_text(
        "Short text",
        tokenizer,
        vocab_embeddings,
        prompt_length,
        rng,
    )
    
    # Check shape
    assert prompt_embeds.shape == (prompt_length, embed_dim)


class TestConfigs:
  """Tests for configuration utilities."""
  
  def test_get_model_config(self):
    """Test getting model configuration."""
    config = configs.get_model_config('2b')
    
    assert config['embed_dim'] == 2048
    assert config['num_layers'] == 26
    assert config['num_heads'] == 16
    assert config['checkpoint'] == 'GEMMA3_2B_IT'
  
  def test_get_model_config_invalid(self):
    """Test error on invalid model size."""
    with pytest.raises(ValueError):
      configs.get_model_config('invalid')
  
  def test_get_prompt_config(self):
    """Test getting prompt configuration."""
    config = configs.get_prompt_config('2b')
    
    assert config['embed_dim'] == 2048
    assert config['prompt_length'] == 20
    assert 'init_scale' in config
  
  def test_get_prompt_config_with_task(self):
    """Test getting task-specific prompt configuration."""
    config = configs.get_prompt_config('2b', task='generation')
    
    assert config['prompt_length'] == 50  # Task-specific
    assert config['embed_dim'] == 2048
  
  def test_get_prompt_config_with_overrides(self):
    """Test configuration with overrides."""
    config = configs.get_prompt_config(
        '2b',
        prompt_length=100,
        init_scale=0.1,
    )
    
    assert config['prompt_length'] == 100  # Overridden
    assert config['init_scale'] == 0.1  # Overridden
    assert config['embed_dim'] == 2048  # From model
  
  def test_get_train_config(self):
    """Test getting training configuration."""
    config = configs.get_train_config()
    
    assert 'learning_rate' in config
    assert 'num_epochs' in config
    assert 'batch_size' in config
  
  def test_get_full_config(self):
    """Test getting complete configuration."""
    config = configs.get_full_config('2b', task='classification')
    
    assert 'model' in config
    assert 'prompt' in config
    assert 'train' in config
    
    assert config['model']['embed_dim'] == 2048
    assert config['prompt']['prompt_length'] == 20
    assert config['train']['learning_rate'] == 0.3


if __name__ == '__main__':
  # Run tests
  pytest.main([__file__, '-v'])
