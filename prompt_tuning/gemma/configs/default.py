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

"""Default configurations for Gemma prompt tuning."""

from typing import Dict, Any


# Model configurations for different Gemma sizes
GEMMA_CONFIGS = {
    '270m': {
        'embed_dim': 1408,
        'num_layers': 18,
        'num_heads': 11,
        'checkpoint': 'GEMMA3_270M_IT',
    },
    '2b': {
        'embed_dim': 2048,
        'num_layers': 26,
        'num_heads': 16,
        'checkpoint': 'GEMMA3_2B_IT',
    },
    '4b': {
        'embed_dim': 2816,
        'num_layers': 32,
        'num_heads': 22,
        'checkpoint': 'GEMMA3_4B_IT',
    },
    '7b': {
        'embed_dim': 3072,
        'num_layers': 28,
        'num_heads': 16,
        'checkpoint': 'GEMMA2_7B_IT',
    },
    '27b': {
        'embed_dim': 4608,
        'num_layers': 46,
        'num_heads': 32,
        'checkpoint': 'GEMMA2_27B_IT',
    },
}


# Default prompt tuning configurations
DEFAULT_PROMPT_CONFIG = {
    'prompt_length': 20,
    'init_scale': 0.5,
    'init_from_vocab': False,
    'init_text': None,
}


# Default training configurations
DEFAULT_TRAIN_CONFIG = {
    'learning_rate': 0.3,
    'weight_decay': 0.0,
    'num_epochs': 3,
    'batch_size': 8,
    'eval_every': 1,
    'save_every': 1,
}


# Task-specific configurations
TASK_CONFIGS = {
    'classification': {
        'prompt_length': 20,
        'learning_rate': 0.3,
        'num_epochs': 5,
    },
    'generation': {
        'prompt_length': 50,
        'learning_rate': 0.1,
        'num_epochs': 3,
    },
    'qa': {
        'prompt_length': 30,
        'learning_rate': 0.2,
        'num_epochs': 5,
    },
    'summarization': {
        'prompt_length': 100,
        'learning_rate': 0.1,
        'num_epochs': 3,
    },
}


def get_model_config(model_size: str) -> Dict[str, Any]:
  """Get configuration for a specific Gemma model size.
  
  Args:
    model_size: Model size ('270m', '2b', '4b', '7b', '27b').
    
  Returns:
    Model configuration dictionary.
  """
  if model_size not in GEMMA_CONFIGS:
    raise ValueError(
        f"Unknown model size: {model_size}. "
        f"Available: {list(GEMMA_CONFIGS.keys())}"
    )
  return GEMMA_CONFIGS[model_size].copy()


def get_prompt_config(
    model_size: str,
    task: str = None,
    **overrides,
) -> Dict[str, Any]:
  """Get prompt configuration for a model and task.
  
  Args:
    model_size: Model size ('270m', '2b', '4b', '7b', '27b').
    task: Optional task name for task-specific config.
    **overrides: Override specific config values.
    
  Returns:
    Prompt configuration dictionary.
  """
  # Start with default config
  config = DEFAULT_PROMPT_CONFIG.copy()
  
  # Add model-specific embed_dim
  model_config = get_model_config(model_size)
  config['embed_dim'] = model_config['embed_dim']
  
  # Apply task-specific config if provided
  if task and task in TASK_CONFIGS:
    config.update(TASK_CONFIGS[task])
  
  # Apply overrides
  config.update(overrides)
  
  return config


def get_train_config(task: str = None, **overrides) -> Dict[str, Any]:
  """Get training configuration.
  
  Args:
    task: Optional task name for task-specific config.
    **overrides: Override specific config values.
    
  Returns:
    Training configuration dictionary.
  """
  # Start with default config
  config = DEFAULT_TRAIN_CONFIG.copy()
  
  # Apply task-specific config if provided
  if task and task in TASK_CONFIGS:
    task_config = TASK_CONFIGS[task]
    for key in ['learning_rate', 'num_epochs']:
      if key in task_config:
        config[key] = task_config[key]
  
  # Apply overrides
  config.update(overrides)
  
  return config


def get_full_config(
    model_size: str,
    task: str = None,
    **overrides,
) -> Dict[str, Any]:
  """Get complete configuration for model, prompt, and training.
  
  Args:
    model_size: Model size ('270m', '2b', '4b', '7b', '27b').
    task: Optional task name for task-specific config.
    **overrides: Override specific config values.
    
  Returns:
    Complete configuration dictionary.
  """
  return {
      'model': get_model_config(model_size),
      'prompt': get_prompt_config(model_size, task, **overrides),
      'train': get_train_config(task, **overrides),
  }
