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
"""Guards authenticate with the caller's token when the configured DataRobot key is dead.

The server runs with an invalid ``DATAROBOT_API_TOKEN`` and ``allow_external_access_tokens``
on (case ``custom-model-external-access-token``). Reaching any test proves startup survived the
dead key. A real DataRobot token stands in for the caller's Okta token in
``x-datarobot-external-access-token``; DataRobot accepts both the same way.

A model guard whose call fails is skipped (fail-open, ``action: report``), so its prediction is
missing from ``datarobot_moderations``. Its presence proves the call authenticated.
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from dotenv import dotenv_values

from dragent_tests.helpers import AGENT
from dragent_tests.helpers import GENERATE_PATH
from dragent_tests.helpers import GENERATE_STREAM_PATH
from dragent_tests.helpers import make_generate_payload
from dragent_tests.helpers import parse_sse_responses

if "external-access-token" not in os.environ.get("WORKFLOW_FILE", ""):
    pytest.skip(
        "Requires the custom-model-external-access-token moderation workflow.",
        allow_module_level=True,
    )

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
# Set by the DataRobot API gateway in production; the test plays the gateway.
EXTERNAL_ACCESS_TOKEN_HEADER = "x-datarobot-external-access-token"
# Stage prefix + the Emotion Classifier's ``target_name`` in the workflow override. The response
# stage runs in post_invoke / the AG-UI stream, so it covers token forwarding there too.
GUARD_METRICS = ("Prompts_target_PREDICTION", "Responses_target_PREDICTION")
PROMPT = "Say 'hello world' and nothing else."


def _real_api_token() -> str:
    """Return the real key, which the test sends as the gateway would forward the caller's token.

    The case env gives this process the server's invalid ``DATAROBOT_API_TOKEN``. Local runs read
    the real key from ``.env``. CI has no ``.env``; its pytest step sets the real key instead.
    """
    token = dotenv_values(ENV_FILE).get("DATAROBOT_API_TOKEN") or os.environ.get(
        "DATAROBOT_API_TOKEN"
    )
    if not token:
        pytest.fail("Needs a real DATAROBOT_API_TOKEN in e2e-tests/.env (or the CI test step).")
    return token


def _moderations(http_client: httpx.Client, path: str, headers: dict[str, str]) -> dict:  # type: ignore[type-arg]
    """POST ``PROMPT`` to ``path`` and return the merged ``datarobot_moderations`` payloads."""
    if path == GENERATE_PATH:
        response = http_client.post(path, json=make_generate_payload(PROMPT), headers=headers)
        assert response.status_code == 200, response.text
        chunks = [DRAgentEventResponse.model_validate_json(response.text)]
    else:
        with http_client.stream(
            "POST", path, json=make_generate_payload(PROMPT), headers=headers
        ) as response:
            assert response.status_code == 200
            chunks = parse_sse_responses(response)
    return {k: v for chunk in chunks for k, v in (chunk.datarobot_moderations or {}).items()}


@pytest.fixture(
    params=[
        GENERATE_STREAM_PATH,
        pytest.param(
            GENERATE_PATH,
            marks=pytest.mark.skipif(
                AGENT == "nat", reason="NAT uses chat-completions format on /generate."
            ),
        ),
    ],
    ids=["stream", "generate"],
)
def path(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.mark.parametrize(
    "headers",
    [{}, {EXTERNAL_ACCESS_TOKEN_HEADER: "invalid-caller-token"}],
    ids=["no-header", "invalid-header"],
)
def test_guard_fails_open_without_valid_token(
    http_client: httpx.Client, path: str, headers: dict[str, str]
) -> None:
    """Without a valid token the guard call fails, but the request still succeeds.

    ``no-header`` falls back to the configured key, so this also proves the server really runs
    with a dead key. ``invalid-header`` shows a bad caller token is not rescued by anything else.
    """
    moderations = _moderations(http_client, path, headers)

    succeeded = [metric for metric in GUARD_METRICS if metric in moderations]
    assert not succeeded, (
        f"Guard unexpectedly succeeded without a valid token: {succeeded}; "
        f"got {sorted(moderations)}"
    )


def test_guard_authenticates_with_caller_token(http_client: httpx.Client, path: str) -> None:
    """A valid caller token in the header makes the guard call succeed."""
    moderations = _moderations(http_client, path, {EXTERNAL_ACCESS_TOKEN_HEADER: _real_api_token()})

    missing = [metric for metric in GUARD_METRICS if metric not in moderations]
    assert not missing, (
        f"Expected {missing} from the guard authenticated by the caller token; "
        f"got {sorted(moderations)}"
    )


def test_caller_token_does_not_leak_into_next_request(http_client: httpx.Client, path: str) -> None:
    """A request without a token must not reuse the previous request's caller token.

    Per-user NAT workflows replay the context of the request that built them; dragent drops the
    per-request part. The keep-alive client sends both requests to the same server worker.
    """
    _moderations(http_client, path, {EXTERNAL_ACCESS_TOKEN_HEADER: _real_api_token()})

    moderations = _moderations(http_client, path, {})

    leaked = [metric for metric in GUARD_METRICS if metric in moderations]
    assert not leaked, f"Guard reused the previous request's token: {leaked}"
