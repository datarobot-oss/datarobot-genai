# Copyright 2025 DataRobot, Inc. and its affiliates.
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
"""Reusable agent utilities and base classes for end-user templates.

This package provides:
- BaseAgent: common initialization for agent env/config fields
- Common helpers: make_system_prompt, extract_user_prompt_content
- Framework utilities (optional extras):
  - crewai: create_pipeline_interactions_from_messages
  - langgraph: create_pipeline_interactions_from_events
  - llamaindex: DataRobotLiteLLM, create_pipeline_interactions_from_events
"""

from ..mcp.common import MCPConfig
from .base import BaseAgent
from .base import InvokeReturn
from .base import UsageMetrics
from .base import default_usage_metrics
from .base import extract_user_prompt_content
from .base import is_streaming
from .base import make_system_prompt

__all__ = [
    "BaseAgent",
    "make_system_prompt",
    "extract_user_prompt_content",
    "default_usage_metrics",
    "is_streaming",
    "InvokeReturn",
    "UsageMetrics",
    "MCPConfig",
]
