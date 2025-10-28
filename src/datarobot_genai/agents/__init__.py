"""Reusable agent utilities and base classes for end-user templates.

This package provides:
- BaseAgent: common initialization for agent env/config fields
- Common helpers: make_system_prompt, choose_model, extract_user_prompt_content
- Framework utilities (optional extras):
  - langgraph: build_chat_llm, create_pipeline_interactions_from_events
  - llamaindex: DataRobotLiteLLM, create_pipeline_interactions_from_events
  - crewai: build_llm, create_pipeline_interactions_from_messages
"""

from .common import BaseAgent
from .common import choose_model
from .common import extract_user_prompt_content
from .common import make_system_prompt

__all__ = [
    "BaseAgent",
    "make_system_prompt",
    "choose_model",
    "extract_user_prompt_content",
]
