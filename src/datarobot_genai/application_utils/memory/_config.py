# Copyright 2026 DataRobot, Inc. and its affiliates.
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

"""Environment-variable resolution and base-URL construction for the Memory Service client."""

from __future__ import annotations

import os


def resolve_endpoint(endpoint: str | None = None) -> str:
    """Resolve the DataRobot API endpoint.

    Priority: explicit argument > ``DATAROBOT_ENDPOINT`` env var.

    Parameters
    ----------
    endpoint : str | None
        Explicit endpoint URL.  If ``None``, the ``DATAROBOT_ENDPOINT``
        environment variable is used.

    Returns
    -------
    str
        The resolved endpoint, with trailing slashes stripped.

    Raises
    ------
    ValueError
        If neither the argument nor the environment variable is set.

    Examples
    --------
    >>> import os; os.environ["DATAROBOT_ENDPOINT"] = "https://app.datarobot.com/api/v2"
    >>> resolve_endpoint()
    'https://app.datarobot.com/api/v2'
    >>> resolve_endpoint("https://my-dr.example.com/api/v2")
    'https://my-dr.example.com/api/v2'
    """
    value = endpoint or os.environ.get("DATAROBOT_ENDPOINT")
    if not value:
        raise ValueError(
            "Missing DataRobot endpoint. Set the DATAROBOT_ENDPOINT environment "
            "variable or pass endpoint= explicitly to MemoryServiceClient."
        )
    return value.rstrip("/")


def resolve_api_token(api_token: str | None = None) -> str:
    """Resolve the DataRobot API token.

    Priority: explicit argument > ``DATAROBOT_API_TOKEN`` env var.

    Parameters
    ----------
    api_token : str | None
        Explicit API token.  If ``None``, the ``DATAROBOT_API_TOKEN``
        environment variable is used.

    Returns
    -------
    str
        The resolved API token.

    Raises
    ------
    ValueError
        If neither the argument nor the environment variable is set.
    """
    value = api_token or os.environ.get("DATAROBOT_API_TOKEN")
    if not value:
        raise ValueError(
            "Missing DataRobot API token. Set the DATAROBOT_API_TOKEN environment "
            "variable or pass api_token= explicitly to MemoryServiceClient."
        )
    return value


def build_base_url(endpoint: str, base_path: str = "memory") -> str:
    """Build the Memory Service base URL from the endpoint and mount path.

    The DataRobot convention is that ``DATAROBOT_ENDPOINT`` already contains
    the ``/api/v2`` suffix.  The Memory Service gateway mounts at
    ``/api/v2/memory``, so the default ``base_path`` is ``"memory"``.

    Parameters
    ----------
    endpoint : str
        DataRobot API endpoint (e.g. ``https://app.datarobot.com/api/v2``).
        Trailing slashes are stripped.
    base_path : str
        Sub-path appended to the endpoint.  Defaults to ``"memory"``.

    Returns
    -------
    str
        Full Memory Service base URL (e.g.
        ``https://app.datarobot.com/api/v2/memory``).

    Examples
    --------
    >>> build_base_url("https://app.datarobot.com/api/v2")
    'https://app.datarobot.com/api/v2/memory'
    >>> build_base_url("https://app.datarobot.com/api/v2/", "memory")
    'https://app.datarobot.com/api/v2/memory'
    """
    clean_endpoint = endpoint.rstrip("/")
    clean_path = base_path.strip("/")
    return f"{clean_endpoint}/{clean_path}"
