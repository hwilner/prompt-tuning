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

"""Experimental prompt tuning techniques.

This module contains experimental implementations of advanced prompt tuning
methods that extend the core prompt tuning functionality.
"""

from prompt_tuning.experimental import p_tuning_v2
from prompt_tuning.experimental import multitask_prompts
from prompt_tuning.experimental import dynamic_prompts
from prompt_tuning.experimental import visual_prompts

__all__ = [
    'p_tuning_v2',
    'multitask_prompts',
    'dynamic_prompts',
    'visual_prompts',
]
