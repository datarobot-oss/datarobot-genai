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

from __future__ import annotations

import abc
import json
import os
from collections.abc import AsyncGenerator
from collections.abc import Mapping
from typing import Any
from typing import Generic
from typing import TypeAlias
from typing import TypedDict
from typing import TypeVar
from typing import cast

from ag_ui.core import Event
from openai.types.chat import CompletionCreateParams
from ragas import MultiTurnSample

from datarobot_genai.core.utils.urls import get_api_base

TTool = TypeVar("TTool")


class BaseAgent(Generic[TTool], abc.ABC):
    """BaseAgent centralizes common initialization for agent templates.

    Fields:
      - api_key: DataRobot API token
      - api_base: Endpoint for DataRobot, normalized for LLM Gateway usage
      - model: Preferred model name
      - timeout: Request timeout
      - verbose: Verbosity flag
      - authorization_context: Authorization context for downstream agents/tools
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        verbose: bool | str | None = True,
        timeout: int | None = 90,
        authorization_context: dict[str, Any] | None = None,
        forwarded_headers: dict[str, str] | None = None,
        **_: Any,
    ) -> None:
        self.api_key = api_key or os.environ.get("DATAROBOT_API_TOKEN")
        self.api_base = (
            api_base or os.environ.get("DATAROBOT_ENDPOINT") or "https://app.datarobot.com"
        )
        self.model = model
        self.timeout = timeout if timeout is not None else 90
        if isinstance(verbose, str):
            self.verbose = verbose.lower() == "true"
        elif verbose is None:
            self.verbose = True
        else:
            self.verbose = bool(verbose)
        self._mcp_tools: list[TTool] = []
        self._authorization_context = authorization_context or {}
        self._forwarded_headers: dict[str, str] = forwarded_headers or {}

    def set_mcp_tools(self, tools: list[TTool]) -> None:
        self._mcp_tools = tools

    @property
    def mcp_tools(self) -> list[TTool]:
        """Return the list of MCP tools available to this agent.

        Subclasses can use this to wire tools into CrewAI agents/tasks during
        workflow construction inside ``build_crewai_workflow``.
        """
        return self._mcp_tools

    @property
    def authorization_context(self) -> dict[str, Any]:
        """Return the authorization context for this agent."""
        return self._authorization_context

    @property
    def forwarded_headers(self) -> dict[str, str]:
        """Return the forwarded headers for this agent."""
        return self._forwarded_headers

    def litellm_api_base(self, deployment_id: str | None) -> str:
        return get_api_base(self.api_base, deployment_id)

    @abc.abstractmethod
    async def invoke(self, completion_create_params: CompletionCreateParams) -> InvokeReturn:
        raise NotImplementedError("Not implemented")

    @classmethod
    def create_pipeline_interactions_from_events(
        cls,
        events: list[Any] | None,
    ) -> MultiTurnSample | None:
        """Create a simple MultiTurnSample from a list of generic events/messages."""
        if not events:
            return None
        return MultiTurnSample(user_input=events)


def extract_user_prompt_content(
    completion_create_params: CompletionCreateParams | Mapping[str, Any],
) -> Any:
    """Extract first user message content from OpenAI messages."""
    params = cast(Mapping[str, Any], completion_create_params)
    user_messages = [msg for msg in params.get("messages", []) if msg.get("role") == "user"]
    # Get the last user message
    user_prompt = user_messages[-1] if user_messages else {}
    content = user_prompt.get("content", {})
    # Try converting prompt from json to a dict
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            pass

    return content


def make_system_prompt(suffix: str = "", *, prefix: str | None = None) -> str:
    """Build a system prompt with optional prefix and suffix.

    Parameters
    ----------
    suffix : str, default ""
        Text appended after the prefix. If non-empty, it is placed on a new line.
    prefix : str | None, keyword-only, default None
        Custom prefix text. When ``None``, a default collaborative assistant
        instruction is used.

    Returns
    -------
    str
        The composed system prompt string.
    """
    default_prefix = (
        "You are a helpful AI assistant, collaborating with other assistants."
        " Use the provided tools to progress towards answering the question."
        " If you are unable to fully answer, that's OK, another assistant with different tools "
        " will help where you left off. Execute what you can to make progress."
    )
    head = prefix if prefix is not None else default_prefix
    if suffix:
        return head + "\n" + suffix
    return head


# Structured type for token usage metrics in responses
class UsageMetrics(TypedDict):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


# Canonical return type for DRUM-compatible invoke implementations
InvokeReturn: TypeAlias = (
    AsyncGenerator[tuple[str | Event, MultiTurnSample | None, UsageMetrics], None]
    | tuple[str, MultiTurnSample | None, UsageMetrics]
)


def default_usage_metrics() -> UsageMetrics:
    """Return a metrics dict with required keys for OpenAI-compatible responses."""
    return {
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
    }


def is_streaming(completion_create_params: CompletionCreateParams | Mapping[str, Any]) -> bool:
    """Return True when the request asks for streaming, False otherwise.

    Accepts both pydantic types and plain dictionaries.
    """
    params = cast(Mapping[str, Any], completion_create_params)
    value = params.get("stream", False)
    # Handle non-bool truthy values defensively (e.g., "true")
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)
