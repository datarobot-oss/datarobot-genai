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

import pytest
from openai import OpenAI

from dragent_tests.helpers import AGENT
from dragent_tests.helpers import BASE_URL


@pytest.mark.parametrize("stream", [True, False])
def test_chat_completions_openai_client(authorization_context_encoded: str, stream: bool) -> None:
    """Call OpenAI-compatible `POST /v1/chat/completions` via the official OpenAI Python client."""
    if AGENT == "nat" and stream:
        pytest.skip(
            "NAT does not support chat completions with streaming: its output is not in the Chat "
            "Completions format"
        )

    # GIVEN: an OpenAI client pointed at the agent's base endpoint with DataRobot auth headers
    client = OpenAI(
        base_url=BASE_URL,
        api_key="e2e",
        default_headers={
            "X-DataRobot-Authorization-Context": authorization_context_encoded,
        },
        timeout=60.0,
    )

    # WHEN: chat completion is requested
    response = client.chat.completions.create(
        model="datarobot-e2e",
        messages=[{"role": "user", "content": "Say 'hello world' and nothing else."}],
        stream=stream,
    )

    if stream:
        full_response = ""
        content_chunk_models: set[str] = set()
        for chunk in response:
            assert chunk.choices
            content = chunk.choices[0].delta.content
            tool_calls = chunk.choices[0].delta.tool_calls
            assert content is not None or tool_calls is not None, "Expected content or tool calls"
            if content:
                content_chunk_models.add(chunk.model)
                full_response += content

        assert len(full_response) > 0, "Expected non-empty assistant message content"
        # content chunks echo the requested model, not NAT's "unknown-model".
        assert content_chunk_models == {"datarobot-e2e"}, (
            f"streaming chunks must echo the requested model, saw {content_chunk_models}"
        )
    else:
        # THEN: the response follows the Chat Completions shape with non-empty assistant text
        assert response.choices
        content = response.choices[0].message.content
        assert content is not None
        assert len(content) > 0, "Expected non-empty assistant message content"
        # the requested model is echoed back (parity with the DRUM baseline).
        assert response.model == "datarobot-e2e"
