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

"""Prompt tuning for LLaMA-2 models.

This package provides prompt tuning capabilities for LLaMA-2 decoder-only
transformer models.
"""

from prompt_tuning.llama2 import prompts
from prompt_tuning.llama2.prompts import (
    HuggingFaceFlaxAdapter,
    LLaMA2JaxAdapter,
    PromptConfig,
    PromptLLaMA,
    SoftPrompt,
    create_prompt_model,
    freeze_base_model,
    get_trainable_params,
    init_from_text,
    init_from_vocab,
)

__all__ = [
    'prompts',
    'HuggingFaceFlaxAdapter',
    'LLaMA2JaxAdapter',
    'PromptConfig',
    'PromptLLaMA',
    'SoftPrompt',
    'create_prompt_model',
    'freeze_base_model',
    'get_trainable_params',
    'init_from_text',
    'init_from_vocab',
]
