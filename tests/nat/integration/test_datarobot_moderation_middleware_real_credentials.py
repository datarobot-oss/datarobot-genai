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

"""Moderation middleware integration tests that call a real DataRobot API.

These mirror the ``*_executes_real_moderations`` tests in
``tests/nat/test_datarobot_moderation_middleware.py`` but omit the credential
verification patch so ``datarobot_dome`` validates ``DATAROBOT_API_TOKEN`` against
``DATAROBOT_ENDPOINT``.

Load credentials from the repository root ``.env`` and/or ``tests/nat/integration/.env``
(``python-dotenv``, ``override=False``). If either variable is missing or the token is
the MCP stub token, tests skip so CI and default ``task test`` runs stay green.

A single ``moderation_config.yaml`` under
``tests/nat/integration/fixtures/moderation_real_credentials/`` combines token / ROUGE / cost
guards with an OOTB LLM-gateway guideline-adherence guard and a faithfulness guard (no
citations in this test path). The
integration tests below assert deterministic token/cost metrics and LLM-gateway score columns
when a real DataRobot endpoint and token are available. Catalog:
`GUARDRAILS.md <https://github.com/datarobot/moderations/blob/main/docs/GUARDRAILS.md>`_.

Edit ``llm_gateway_model_id`` on each LLM guard in that YAML if your tenant uses a different
gateway model slug (defaults follow ``LLM_DEFAULT_MODEL`` in ``tests/nat/integration/.env.sample``).
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from dotenv import load_dotenv
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import Message as NATAPIMessage
from nat.data_models.api_server import Usage as NATChatUsage

pytest.importorskip("datarobot_dome")

from datarobot_dome.constants import AGENT_GOAL_ACCURACY_COLUMN_NAME
from datarobot_dome.constants import FAITHFULLNESS_COLUMN_NAME
from datarobot_dome.constants import GUIDELINE_ADHERENCE_COLUMN_NAME
from datarobot_dome.constants import ROUGE_1_COLUMN_NAME
from datarobot_dome.constants import TASK_ADHERENCE_SCORE_COLUMN_NAME

from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.nat.datarobot_moderation_middleware import DataRobotModerationConfig
from datarobot_genai.nat.datarobot_moderation_middleware import DataRobotModerationMiddleware
from tests.drmcp.stub_credentials import STUB_DATAROBOT_API_TOKEN
from tests.nat.test_datarobot_moderation_middleware import _fn_context
from tests.nat.test_datarobot_moderation_middleware import _make_run_input
from tests.nat.test_datarobot_moderation_middleware import _moderation_config_from_fixture_dir
from tests.nat.test_datarobot_moderation_middleware import _nat_chat_response_assistant_text
from tests.nat.test_datarobot_moderation_middleware import _text_response


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _integration_dir() -> Path:
    return Path(__file__).resolve().parent


REAL_CREDENTIALS_MODERATION_MODEL_DIR = (
    _integration_dir() / "fixtures" / "moderation_real_credentials"
)


def _build_moderation_middleware_for_model_dir(
    *,
    model_dir: Path,
    builder_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> DataRobotModerationMiddleware:
    monkeypatch.setenv("TARGET_NAME", '"response"')
    try:
        mw = DataRobotModerationMiddleware(
            DataRobotModerationConfig(moderation=_moderation_config_from_fixture_dir(model_dir)),
            builder_mock,
        )
    except RuntimeError as exc:
        if "cannot reach" in str(exc):
            pytest.skip(str(exc))
        raise
    assert mw.enabled is True
    return mw


@pytest.fixture(scope="module")
def _load_dotenv_for_local_datarobot() -> None:
    # Repo root first, then integration-local .env for keys still unset.
    load_dotenv(_repo_root() / ".env", override=False)
    load_dotenv(_integration_dir() / ".env", override=False)


@pytest.fixture
def real_datarobot_credentials(_load_dotenv_for_local_datarobot: None) -> None:
    token = (os.environ.get("DATAROBOT_API_TOKEN") or "").strip()
    endpoint = (os.environ.get("DATAROBOT_ENDPOINT") or "").strip()
    if not token or not endpoint:
        pytest.skip(
            "Set DATAROBOT_API_TOKEN and DATAROBOT_ENDPOINT in the repository .env "
            "or tests/nat/integration/.env to run real-credential moderation tests."
        )
    if token == STUB_DATAROBOT_API_TOKEN:
        pytest.skip("DATAROBOT_API_TOKEN is the stub token; use a real API token for these tests.")


@pytest.fixture
def builder_mock() -> MagicMock:
    return MagicMock()


@pytest.fixture
def moderation_middleware_with_env(
    real_datarobot_credentials: None,
    builder_mock: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> DataRobotModerationMiddleware:
    return _build_moderation_middleware_for_model_dir(
        model_dir=REAL_CREDENTIALS_MODERATION_MODEL_DIR,
        builder_mock=builder_mock,
        monkeypatch=monkeypatch,
    )


async def test_function_middleware_invoke_integration_executes_real_moderations(
    moderation_middleware_with_env: DataRobotModerationMiddleware,
) -> None:
    mw = moderation_middleware_with_env

    async def call_next(*_a: Any, **_k: Any) -> DRAgentEventResponse:
        return _text_response("This is a test response.")

    result = await mw.function_middleware_invoke(
        _make_run_input("Count moderation tokens for this prompt."),
        call_next=call_next,
        context=_fn_context(),
    )

    assert isinstance(result, DRAgentEventResponse)
    assert result.datarobot_moderations is not None
    mods = result.datarobot_moderations
    assert mods["Prompts_token_count"] == 7
    assert mods["Responses_token_count"] == 6
    assert mods["cost"] == pytest.approx(0.019)
    assert mods["prompt_token_count_from_usage"] == mods["Prompts_token_count"]
    assert mods["response_token_count_from_usage"] == mods["Responses_token_count"]
    assert GUIDELINE_ADHERENCE_COLUMN_NAME in mods
    assert TASK_ADHERENCE_SCORE_COLUMN_NAME in mods
    assert AGENT_GOAL_ACCURACY_COLUMN_NAME in mods
    assert ROUGE_1_COLUMN_NAME not in mods  # No citations
    assert FAITHFULLNESS_COLUMN_NAME not in mods  # No citations


async def test_function_middleware_invoke_integration_nat_chat_input_chat_response_real_moderations(
    moderation_middleware_with_env: DataRobotModerationMiddleware,
) -> None:
    mw = moderation_middleware_with_env
    crm = ChatRequestOrMessage(
        messages=[
            NATAPIMessage(role="user", content="Count moderation tokens for this prompt."),
        ],
    )

    async def call_next(*_a: Any, **_k: Any) -> ChatResponse:
        return _nat_chat_response_assistant_text("This is a test response.")

    result = await mw.function_middleware_invoke(
        crm,
        call_next=call_next,
        context=_fn_context(),
    )

    assert isinstance(result, ChatResponse)
    assert result.choices[0].message.content == "This is a test response."
    assert result.usage == NATChatUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    assert result.datarobot_moderations is not None
    mods = result.datarobot_moderations
    assert mods["Prompts_token_count"] == 7
    assert mods["Responses_token_count"] == 6
    assert mods["cost"] == pytest.approx(0.019)
    assert mods["prompt_token_count_from_usage"] == mods["Prompts_token_count"]
    assert mods["response_token_count_from_usage"] == mods["Responses_token_count"]
    assert GUIDELINE_ADHERENCE_COLUMN_NAME in mods
    assert TASK_ADHERENCE_SCORE_COLUMN_NAME in mods
    assert AGENT_GOAL_ACCURACY_COLUMN_NAME in mods
    assert ROUGE_1_COLUMN_NAME not in mods  # No citations
    assert FAITHFULLNESS_COLUMN_NAME not in mods  # No citations


async def test_function_middleware_stream_integration_executes_real_moderations(
    moderation_middleware_with_env: DataRobotModerationMiddleware,
) -> None:
    mw = moderation_middleware_with_env

    async def upstream() -> AsyncGenerator[Any]:
        yield _text_response("This is a test response.")

    stream_next = MagicMock(return_value=upstream())
    chunks = [
        item
        async for item in mw.function_middleware_stream(
            _make_run_input("Count moderation tokens for this prompt."),
            call_next=stream_next,
            context=_fn_context(),
        )
    ]

    assert chunks
    moderation_payloads = [c.datarobot_moderations for c in chunks if c.datarobot_moderations]
    assert moderation_payloads
    all_keys = {k for payload in moderation_payloads for k in payload}
    assert "Prompts_token_count" in all_keys
    assert "Responses_token_count" in all_keys
    assert "cost" in all_keys
    final_mods = moderation_payloads[-1]
    assert final_mods["Prompts_token_count"] == 7
    assert final_mods["Responses_token_count"] == 6
    assert final_mods["cost"] == pytest.approx(0.019)
    assert GUIDELINE_ADHERENCE_COLUMN_NAME in all_keys
    assert TASK_ADHERENCE_SCORE_COLUMN_NAME in all_keys
    assert AGENT_GOAL_ACCURACY_COLUMN_NAME in all_keys
    assert ROUGE_1_COLUMN_NAME not in all_keys  # No citations
    assert FAITHFULLNESS_COLUMN_NAME not in all_keys  # No citations
