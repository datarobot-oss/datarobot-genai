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
"""Per-request, thread-safe DataRobot REST client.

Lives in ``drtools`` so consumers pinning ``datarobot-genai[drtools]`` (e.g.
global-mcp) and any agent importing drtools directly can call the DataRobot
API as the requesting user without depending on ``drmcp``.

Thread-safety: backed by :func:`datarobot.client.client_configuration`, which
stores the client in a ``ContextVar``. Concurrent asyncio tasks / threads each
get their own client, so we never mutate the process-global ``dr.Client()``
(which would leak one request's token into a concurrent request). See
MODEL-23521.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import datarobot as dr
from datarobot.client import client_configuration
from datarobot.rest import RESTClientObject

from datarobot_genai.drtools.core.auth import resolve_token_from_headers
from datarobot_genai.drtools.core.credentials import get_credentials


def resolve_request_user_token(*, headers_auth_only: bool = False) -> str:
    """Resolve the requesting user's DataRobot API token.

    Resolution order:

    1. Token from request headers (via :func:`resolve_token_from_headers`).
    2. If unset and ``headers_auth_only=False``, the application API token
       from credentials (e.g. dynamic registration / non-HTTP contexts).

    Raises
    ------
    ValueError
        If ``headers_auth_only=True`` and no token is present in the headers.
    """
    token = resolve_token_from_headers()
    if not token:
        if headers_auth_only:
            raise ValueError("No API token found in request headers")
        token = get_credentials().datarobot.application_api_token
    return token


@contextmanager
def request_user_dr_client(*, headers_auth_only: bool = False) -> Iterator[RESTClientObject]:
    """Yield a request-user-scoped ``RESTClientObject`` for the block's duration.

    Use inside a ``with`` so the configured client stays scoped to this task::

        with request_user_dr_client() as client:
            client.post("entitlements/evaluate/", json=...)

    Thread-safe: ``client_configuration()`` is ``ContextVar``-scoped, so this
    does not mutate the global ``dr.Client()`` and will not mix tokens across
    concurrent requests.
    """
    token = resolve_request_user_token(headers_auth_only=headers_auth_only)
    endpoint = get_credentials().datarobot.endpoint
    with client_configuration(token=token, endpoint=endpoint):
        yield cast(RESTClientObject, dr.client.get_client())
