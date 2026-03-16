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

# Load .env from e2e-tests root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@pytest.fixture(scope="session")
def session_secret_key() -> str:
    return os.environ.get("SESSION_SECRET_KEY")


@pytest.fixture(scope="session")
def datarobot_user_id() -> str:
    return os.environ.get("DATAROBOT_USER_ID")


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
def http_client(authorization_context_encoded: str) -> Generator[httpx.Client]:  # type: ignore[type-arg]
    timeout = httpx.Timeout(connect=10, read=300, write=10, pool=10)
    headers = {
        "X-DataRobot-Authorization-Context": authorization_context_encoded,
    }
    with httpx.Client(
        base_url="http://localhost:8080", timeout=timeout, headers=headers
    ) as client:
        yield client
