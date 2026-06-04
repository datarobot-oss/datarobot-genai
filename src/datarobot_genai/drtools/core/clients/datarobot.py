# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-request, thread-safe DataRobot API client for tools.

Thread-safety: backed by :func:`datarobot.client.client_configuration`, which
stores the client in a ``ContextVar``. Concurrent asyncio tasks / threads each
get their own client, so we never mutate the process-global ``dr.Client()``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from typing import cast

import datarobot as dr
from datarobot.client import client_configuration
from datarobot.context import Context as DRContext
from datarobot.rest import RESTClientObject

from datarobot_genai.drtools.core.auth import resolve_token_from_headers
from datarobot_genai.drtools.core.credentials import get_credentials
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)


def get_datarobot_access_token(*, headers_auth_only: bool = True) -> str:
    """Resolve the requesting user's DataRobot API token.

    Resolution order:

    1. Token from request headers (via :func:`resolve_token_from_headers`).
    2. If unset and ``headers_auth_only=False``, the application API token
       from credentials (e.g. dynamic registration / non-HTTP contexts).

    Raises
    ------
    ToolError
        If ``headers_auth_only=True`` and no token is present in the headers.
    """
    token = resolve_token_from_headers()
    if not token:
        if headers_auth_only:
            raise ToolError(
                "DataRobot API token not found in headers. "
                "Please provide it via 'Authorization' (Bearer), 'x-datarobot-api-token' headers.",
                kind=ToolErrorKind.AUTHENTICATION,
            )
        token = get_credentials().datarobot.datarobot_api_token
    return token


@contextmanager
def request_user_dr_client(*, headers_auth_only: bool = True) -> Iterator[RESTClientObject]:
    """Yield a request-user-scoped ``RESTClientObject`` for the block's duration.

    Use inside a ``with`` so the configured client stays scoped to this task::

        with request_user_dr_client() as client:
            client.post("entitlements/evaluate/", json=...)

    Thread-safe: ``client_configuration()`` is ``ContextVar``-scoped, so this
    does not mutate the global ``dr.Client()`` and will not mix tokens across
    concurrent requests.
    """
    token = get_datarobot_access_token(headers_auth_only=headers_auth_only)
    endpoint = get_credentials().datarobot.datarobot_endpoint
    with client_configuration(token=token, endpoint=endpoint):
        # Avoid use-case context from trafaret affecting tool calls.
        DRContext.use_case = None
        yield cast(RESTClientObject, dr.client.get_client())


@contextmanager
def request_user_dr_sdk(*, headers_auth_only: bool = True) -> Iterator[Any]:
    """Yield the ``datarobot`` module with a request-scoped SDK client configured.

    Use for SDK calls (e.g. ``dr.Deployment.get``) inside the ``with`` block::

        with request_user_dr_sdk(headers_auth_only=True):
            deployment = dr.Deployment.get(deployment_id)

    Thread-safe: same :func:`client_configuration` scoping as
    :func:`request_user_dr_client`.
    """
    token = get_datarobot_access_token(headers_auth_only=headers_auth_only)
    endpoint = get_credentials().datarobot.datarobot_endpoint
    with client_configuration(token=token, endpoint=endpoint):
        DRContext.use_case = None
        yield dr


class ThreadSafeDataRobotClient:
    """Configure a per-request DataRobot SDK client from the caller's headers."""

    def __init__(self) -> None:
        self.endpoint = get_credentials().datarobot.datarobot_endpoint

    @contextmanager
    def request_user_client(self, *, headers_auth_only: bool = True) -> Iterator[RESTClientObject]:
        """Yield a scoped REST client; same semantics as :func:`request_user_dr_client`."""
        token = get_datarobot_access_token(headers_auth_only=headers_auth_only)
        with client_configuration(token=token, endpoint=self.endpoint):
            # Avoid use-case context from trafaret affecting tool calls.
            DRContext.use_case = None
            yield cast(RESTClientObject, dr.client.get_client())
