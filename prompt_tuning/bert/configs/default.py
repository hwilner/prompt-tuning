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

"""Default configurations for BERT prompt tuning."""

from typing import Dict, Any


# Model configurations for different BERT variants
BERT_CONFIGS = {
    'bert-base-uncased': {
        'embed_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'vocab_size': 30522,
        'max_position_embeddings': 512,
    },
    'bert-large-uncased': {
        'embed_dim': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'vocab_size': 30522,
        'max_position_embeddings': 512,
    },
    'bert-base-cased': {
        'embed_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'vocab_size': 28996,
        'max_position_embeddings': 512,
    },
    'bert-large-cased': {
        'embed_dim': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'vocab_size': 28996,
        'max_position_embeddings': 512,
    },
    'distilbert-base-uncased': {
        'embed_dim': 768,
        'num_layers': 6,
        'num_heads': 12,
        'vocab_size': 30522,
        'max_position_embeddings': 512,
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
    'weight_decay': 0.01,
    'num_epochs': 5,
    'batch_size': 16,
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
    'ner': {
        'prompt_length': 10,
        'learning_rate': 0.2,
        'num_epochs': 5,
    },
    'qa': {
        'prompt_length': 30,
        'learning_rate': 0.2,
        'num_epochs': 3,
    },
    'sentiment': {
        'prompt_length': 15,
        'learning_rate': 0.3,
        'num_epochs': 3,
    },
}


def get_model_config(model_name: str) -> Dict[str, Any]:
  """Get configuration for a specific BERT model."""
  if model_name not in BERT_CONFIGS:
    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Available: {list(BERT_CONFIGS.keys())}"
    )
  return BERT_CONFIGS[model_name].copy()


def get_prompt_config(
    model_name: str,
    task: str = None,
    **overrides,
) -> Dict[str, Any]:
  """Get prompt configuration for a model and task."""
  config = DEFAULT_PROMPT_CONFIG.copy()
  
  model_config = get_model_config(model_name)
  config['embed_dim'] = model_config['embed_dim']
  
  if task and task in TASK_CONFIGS:
    config.update(TASK_CONFIGS[task])
  
  config.update(overrides)
  
  return config


def get_train_config(task: str = None, **overrides) -> Dict[str, Any]:
  """Get training configuration."""
  config = DEFAULT_TRAIN_CONFIG.copy()
  
  if task and task in TASK_CONFIGS:
    task_config = TASK_CONFIGS[task]
    for key in ['learning_rate', 'num_epochs']:
      if key in task_config:
        config[key] = task_config[key]
  
  config.update(overrides)
  
  return config


def get_full_config(
    model_name: str,
    task: str = None,
    **overrides,
) -> Dict[str, Any]:
  """Get complete configuration for model, prompt, and training."""
  return {
      'model': get_model_config(model_name),
      'prompt': get_prompt_config(model_name, task, **overrides),
      'train': get_train_config(task, **overrides),
  }
