"""Reusable agent utilities and base classes for end-user templates.

This package provides:
- BaseAgent: common initialization for agent env/config fields
- Common helpers: make_system_prompt, extract_user_prompt_content
- Framework utilities (optional extras):
  - crewai: build_llm, create_pipeline_interactions_from_messages
  - langgraph: create_pipeline_interactions_from_events
  - llamaindex: DataRobotLiteLLM, create_pipeline_interactions_from_events
"""

from .common import BaseAgent
from .common import extract_user_prompt_content
from .common import make_system_prompt

__all__ = [
    "BaseAgent",
    "make_system_prompt",
    "extract_user_prompt_content",
]
