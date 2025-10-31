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

import abc
import json
import os
from collections.abc import AsyncGenerator
from collections.abc import Mapping
from typing import Any
from typing import Union
from typing import cast

from openai.types.chat import CompletionCreateParams
from ragas import MultiTurnSample

from datarobot_genai.core.utils.urls import get_api_base


class BaseAgent(abc.ABC):
    """BaseAgent centralizes common initialization for agent templates.

    Fields:
      - api_key: DataRobot API token
      - api_base: Endpoint for DataRobot, normalized for LLM Gateway usage
      - model: Preferred model name
      - timeout: Request timeout
      - verbose: Verbosity flag
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        model: str | None = None,
        verbose: bool | str | None = True,
        timeout: int | None = 90,
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

    def litellm_api_base(self, deployment_id: str | None) -> str:
        return get_api_base(self.api_base, deployment_id)

    @abc.abstractmethod
    async def invoke(
        self, completion_create_params: CompletionCreateParams
    ) -> Union[  # noqa: UP007
        AsyncGenerator[tuple[str, Any | None, dict[str, int]], None],
        tuple[str, Any | None, dict[str, int]],
    ]:
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


# Canonical return type for DRUM-compatible invoke implementations
InvokeReturn = (
    AsyncGenerator[tuple[str, MultiTurnSample | None, dict[str, int]], None]
    | tuple[str, MultiTurnSample | None, dict[str, int]]
)


def default_usage_metrics() -> dict[str, int]:
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
