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

"""Authorization context helpers for chat flows."""

from typing import Any

from datarobot.models.genai.agent.auth import set_authorization_context
from openai.types import CompletionCreateParams
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming
from openai.types.chat.completion_create_params import CompletionCreateParamsStreaming

from datarobot_genai.core.utils.auth import AuthContextHeaderHandler


def _get_authorization_context_from_headers(
    headers: dict[str, str],
    secret_key: str | None = None,
) -> dict[str, Any] | None:
    """Extract authorization context from headers using AuthContextHeaderHandler.

    Parameters
    ----------
    headers : dict[str, str]
        HTTP headers from which to extract the authorization context.
    secret_key : str | None
        Secret key for JWT decoding. If None, retrieves from environment variable.

    Returns
    -------
    dict[str, Any] | None
        The extracted authorization context, or None if not found.
    """
    handler = AuthContextHeaderHandler(secret_key=secret_key)
    if context := handler.get_context(headers):
        return context.model_dump()
    return None


def _get_authorization_context_from_params(
    completion_create_params: CompletionCreateParams
    | CompletionCreateParamsNonStreaming
    | CompletionCreateParamsStreaming,
) -> dict[str, Any] | None:
    """Extract authorization context from completion create parameters.

    Parameters
    ----------
    completion_create_params : CompletionCreateParams
        The parameters used to create the completion.

    Returns
    -------
    dict[str, Any] | None
        The extracted authorization context, or None if not found.
    """
    return completion_create_params.get("authorization_context", None)


def initialize_authorization_context(
    completion_create_params: CompletionCreateParams
    | CompletionCreateParamsNonStreaming
    | CompletionCreateParamsStreaming,
    secret_key: str | None = None,
    **kwargs: Any,
) -> None:
    """Set the authorization context for the agent.

    Authorization context is required for propagating information needed by downstream
    agents and tools to retrieve access tokens to connect to external services. When set,
    authorization context will be propagated when using MCP Server component or when
    using ToolClient class.

    Parameters
    ----------
    completion_create_params : CompletionCreateParams | CompletionCreateParamsNonStreaming |
        CompletionCreateParamsStreaming
        Parameters supplied to the completion API. May include a fallback
        ``authorization_context`` mapping under the same key.
    secret_key : str | None
        Optional secret used to decode a JWT provided in headers. If ``None``,
        the handler may obtain the secret from the environment.
    **kwargs : Any
        Additional keyword arguments. Expected to include a ``headers`` key
        containing incoming HTTP headers as ``dict[str, str]``.

    """
    incoming_headers = kwargs.get("headers", {})

    # Recommended way of propagating authorization context is via headers
    # with JWT endoding/decoding for additional security. The completion params
    # is used as a fallback for backward compatibility only and may be removed in
    # the future.
    authorization_context: dict[str, Any] = (
        _get_authorization_context_from_headers(incoming_headers, secret_key)
        or _get_authorization_context_from_params(completion_create_params)
        or {}
    )

    # Note: authorization context internally uses contextvars, which are
    # thread-safe and async-safe.
    set_authorization_context(authorization_context)
