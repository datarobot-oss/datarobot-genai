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

"""Workload identity tokens for DataRobot workloads.

``get_workload_token_source()`` returns the process-wide ``WorkloadTokenSource``, or
``None`` when the workload has no identity. ``WorkloadTokenSource.token()`` returns
the current token or raises ``WorkloadTokenError``.

The ``requests`` and ``httpx`` auth adapters are in ``.auth``, which is not imported
here, so httpx loads only where it is used. Nothing here imports the rest of
datarobot_genai, so the package can move to a shared library as is.
"""

from .config import ASSERTION_AUDIENCE_ENV
from .config import CLIENT_ID_ENV
from .config import DEFAULT_KEY_ALG
from .config import DEFAULT_REFRESH_INTERVAL_SECONDS
from .config import DISCOVERY_URL_ENV
from .config import KEY_ALG_ENV
from .config import KEY_ID_ENV
from .config import PRIVATE_KEY_ENV
from .config import PUBLIC_API_ENDPOINT_ENV
from .config import REFRESH_INTERVAL_ENV
from .config import TOKEN_ENDPOINT_ENV
from .config import WORKLOAD_ID_ENV
from .config import WORKLOAD_TOKEN_FILE_ENV
from .config import AuthMode
from .config import WorkloadIdentityConfig
from .config import WorkloadTokenError
from .source import WorkloadTokenSource
from .source import get_workload_token_source

__all__ = [
    "ASSERTION_AUDIENCE_ENV",
    "CLIENT_ID_ENV",
    "DEFAULT_KEY_ALG",
    "DEFAULT_REFRESH_INTERVAL_SECONDS",
    "DISCOVERY_URL_ENV",
    "KEY_ALG_ENV",
    "KEY_ID_ENV",
    "PRIVATE_KEY_ENV",
    "PUBLIC_API_ENDPOINT_ENV",
    "REFRESH_INTERVAL_ENV",
    "TOKEN_ENDPOINT_ENV",
    "WORKLOAD_ID_ENV",
    "WORKLOAD_TOKEN_FILE_ENV",
    "AuthMode",
    "WorkloadIdentityConfig",
    "WorkloadTokenError",
    "WorkloadTokenSource",
    "get_workload_token_source",
]
