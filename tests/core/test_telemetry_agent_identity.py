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

from opentelemetry import baggage

from datarobot_genai.core.telemetry.agent_identity import GEN_AI_AGENT_NAME_BAGGAGE_KEY
from datarobot_genai.core.telemetry.agent_identity import agent_name_baggage
from datarobot_genai.core.telemetry.agent_identity import attach_agent_name_baggage
from datarobot_genai.core.telemetry.agent_identity import detach_agent_name_baggage


def test_attach_agent_name_baggage_sets_it_in_the_active_context() -> None:
    token = attach_agent_name_baggage("researcher")
    try:
        assert baggage.get_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY) == "researcher"
    finally:
        detach_agent_name_baggage(token)

    # Detach reverts to no baggage - it doesn't leak past its own scope.
    assert baggage.get_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY) is None


def test_attach_agent_name_baggage_is_a_noop_for_falsy_names() -> None:
    for falsy in (None, ""):
        token = attach_agent_name_baggage(falsy)
        assert token is None
        assert baggage.get_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY) is None


def test_detach_agent_name_baggage_is_a_noop_for_none() -> None:
    # Must not raise - attach_agent_name_baggage(None) returns None, and callers
    # are expected to pass that straight through without special-casing it.
    detach_agent_name_baggage(None)


def test_agent_name_baggage_context_manager_scopes_correctly() -> None:
    assert baggage.get_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY) is None
    with agent_name_baggage("planner"):
        assert baggage.get_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY) == "planner"
    assert baggage.get_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY) is None


def test_agent_name_baggage_context_manager_is_a_noop_without_a_name() -> None:
    with agent_name_baggage(None):
        assert baggage.get_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY) is None


def test_agent_name_baggage_detaches_even_on_exception() -> None:
    class _BoomError(Exception):
        pass

    try:
        with agent_name_baggage("planner"):
            assert baggage.get_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY) == "planner"
            raise _BoomError
    except _BoomError:
        pass

    assert baggage.get_baggage(GEN_AI_AGENT_NAME_BAGGAGE_KEY) is None
