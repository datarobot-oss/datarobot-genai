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

import os
from typing import Any

from ragas import MultiTurnSample

from ..utils.urls import get_api_base


class BaseAgent:
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
        verbose: bool | str = True,
        timeout: int = 90,
        **_: Any,
    ) -> None:
        self.api_key = api_key or os.environ.get("DATAROBOT_API_TOKEN")
        self.api_base = (
            api_base or os.environ.get("DATAROBOT_ENDPOINT") or "https://app.datarobot.com"
        )
        self.model = model
        self.timeout = timeout
        if isinstance(verbose, str):
            self.verbose = verbose.lower() == "true"
        else:
            self.verbose = bool(verbose)

    def litellm_api_base(self, deployment_id: str | None) -> str:
        return get_api_base(self.api_base, deployment_id)


def extract_user_prompt_content(completion_create_params: dict[str, Any]) -> Any:
    """Extract first user message content from OpenAI messages."""
    user_messages = [
        msg for msg in completion_create_params.get("messages", []) if msg.get("role") == "user"
    ]
    user_prompt = user_messages[0] if user_messages else {}
    return user_prompt.get("content", {})


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


def create_pipeline_interactions_from_events_simple(
    events: list[Any] | None,
) -> MultiTurnSample | None:
    """Create a simple MultiTurnSample from a list of generic events/messages."""
    if not events:
        return None
    return MultiTurnSample(user_input=events)
