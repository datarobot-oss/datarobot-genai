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
"""Assert the NeMo moderation guard actually blocks disallowed input.

The other moderation tests only check that ``datarobot_moderations`` metadata is
*present* on an on-topic prompt, so a guard that silently fails open is invisible
to them. This test sends input that the prompt-stage guard must block and asserts
the configured ``block`` intervention message is returned -- turning a
fail-open guard into a red test.

Only runs under the nemo-guardrails moderation workflow: the disallowed input is a
blocked term from ``nemo_guardrails/blocked_terms.txt`` (a deterministic string
match), and the intervention message is defined in
``workflow-nemo-guardrails-moderations.yaml``.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse

from dragent_tests.helpers import AGENT
from dragent_tests.helpers import GENERATE_PATH
from dragent_tests.helpers import collect_text
from dragent_tests.helpers import make_generate_payload

if "nemo-guardrails" not in os.environ.get("WORKFLOW_FILE", ""):
    pytest.skip(
        "Block-intervention test requires the nemo-guardrails moderation workflow "
        "(blocked_terms + 'stay on topic' guards).",
        allow_module_level=True,
    )

if AGENT == "nat":
    pytest.skip(
        "NAT returns a single response in chat-completions format on /generate; "
        "the AG-UI generate path is covered by the other agents.",
        allow_module_level=True,
    )

# Defined as the guard's ``intervention.message`` in
# dragent/*/workflow-nemo-guardrails-moderations.yaml.
BLOCK_MESSAGE = "This topic is outside the allowed scope."

# A blocked term from dragent/*/nemo_guardrails/blocked_terms.txt -- the
# ``check_blocked_terms`` action is a deterministic substring match, so once the
# guard's gateway LLM call succeeds the block fires reliably (no LLM-judgment
# flakiness).
BLOCKED_TERM = "competitor_name_1"


def test_disallowed_input_is_blocked(http_client: httpx.Client) -> None:
    """The prompt-stage guard blocks disallowed input and returns the block message."""
    # GIVEN: a prompt containing a blocked term
    payload = make_generate_payload(f"How does {BLOCKED_TERM} compare to your product?")

    # WHEN: the prompt is sent to the non-streaming generate endpoint
    response = http_client.post(GENERATE_PATH, json=payload)
    assert response.status_code == 200
    response_data = DRAgentEventResponse.model_validate_json(response.text)

    # THEN: moderation metadata is attached (guards ran)
    assert response_data.datarobot_moderations, (
        "Expected datarobot_moderations on the response when guards are configured; "
        "its absence means the guard pipeline did not run."
    )

    # THEN: the guard intervened -- the configured block message is returned instead
    # of an agent answer. Accept the message in the response text or the serialized
    # moderation payload, so the assertion does not over-fit dome's surfacing.
    full_text = collect_text(response_data.events)
    moderations_blob = json.dumps(response_data.datarobot_moderations)
    assert BLOCK_MESSAGE in full_text or BLOCK_MESSAGE in moderations_blob, (
        "Expected the moderation guard to BLOCK the disallowed input and return "
        f"{BLOCK_MESSAGE!r}. The guard appears to have failed open. "
        f"text={full_text!r} moderations={response_data.datarobot_moderations!r}"
    )
