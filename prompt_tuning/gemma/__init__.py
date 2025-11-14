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

"""Prompt tuning for Gemma models.

This package provides prompt tuning capabilities for Gemma decoder-only
transformer models. It includes:

- Soft prompt modules
- Training utilities
- Configuration management
- Example scripts

Example:
  ```python
  from gemma import gm
  from prompt_tuning.gemma import prompts, train
  
  # Create prompt-tuned model
  model = gm.nn.Gemma3_2B()
  prompt_config = prompts.PromptConfig(prompt_length=20, embed_dim=2048)
  prompt_model = prompts.create_prompt_model(model, prompt_config)
  
  # Train
  trained_params = train.train_prompt(
      prompt_model,
      train_dataset,
      num_steps=1000,
  )
  ```
"""

from prompt_tuning.gemma import prompts
from prompt_tuning.gemma.prompts import (
    PromptConfig,
    PromptGemma,
    SoftPrompt,
    create_prompt_model,
    freeze_base_model,
    get_trainable_params,
    init_from_text,
    init_from_vocab,
)

__all__ = [
    'prompts',
    'PromptConfig',
    'PromptGemma',
    'SoftPrompt',
    'create_prompt_model',
    'freeze_base_model',
    'get_trainable_params',
    'init_from_text',
    'init_from_vocab',
]
