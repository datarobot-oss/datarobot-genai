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

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest
from datarobot.auth.session import AuthCtx
from datarobot.auth.users import User
from datarobot_genai.core.utils.auth import AuthContextHeaderHandler
from dotenv import load_dotenv

from .helpers import BASE_URL
from .helpers import MOCK_OTEL_COLLECTOR_PORT
from .mock_otel_collector import MockOtelCollector

# Load .env from e2e-tests root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@pytest.fixture(scope="session", autouse=True)
def otel_collector() -> Generator[MockOtelCollector]:  # type: ignore[type-arg]
    """In-process OTLP collector the dragent server exports spans to.

    Bound to a fixed port (matching the server's ``OTEL_EXPORTER_OTLP_ENDPOINT``
    in ``dragent/Taskfile.yaml``) and kept up for the whole session so any test
    can verify exported spans via ``helpers.assert_tracing_conventions``.
    Autouse so it is always listening, even for tests that don't assert on
    traces — otherwise the server's exports would just be dropped.
    """
    with MockOtelCollector(port=MOCK_OTEL_COLLECTOR_PORT) as collector:
        yield collector


@pytest.fixture(scope="session")
def session_secret_key() -> str:
    return os.environ["SESSION_SECRET_KEY"]


@pytest.fixture(scope="session")
def datarobot_user_id() -> str:
    return os.environ["DATAROBOT_USER_ID"]


@pytest.fixture(scope="session")
def authorization_context_encoded(session_secret_key, datarobot_user_id) -> str:
    ctx = AuthCtx(
        user=User(
            id=datarobot_user_id,
            name="buzok-ci-agents",
            email="buzok-ci-agents@datarobot.com",
        ),
        identities=[]
    )

    return AuthContextHeaderHandler(secret_key=session_secret_key).encode(ctx.model_dump())


@pytest.fixture(scope="session")
def http_client(authorization_context_encoded: str, datarobot_user_id: str) -> Generator[httpx.Client]:  # type: ignore[type-arg]
    timeout = httpx.Timeout(connect=10, read=120, write=10, pool=10)
    headers = {
        "X-DataRobot-Authorization-Context": authorization_context_encoded,
        "X-DataRobot-User-Id": datarobot_user_id,
    }
    with httpx.Client(
        base_url=BASE_URL, timeout=timeout, headers=headers
    ) as client:
        yield client


@pytest.fixture(scope="session")
def gateway_http_client(datarobot_user_id: str) -> Generator[httpx.Client]:  # type: ignore[type-arg]
    """HTTP client with only ``X-DataRobot-User-Id`` -- simulates direct API
    calls through the gateway where no signed auth-context JWT is present.
    """
    timeout = httpx.Timeout(connect=10, read=120, write=10, pool=10)
    headers = {
        "X-DataRobot-User-Id": datarobot_user_id,
    }
    with httpx.Client(
        base_url=BASE_URL, timeout=timeout, headers=headers
    ) as client:
        yield client
