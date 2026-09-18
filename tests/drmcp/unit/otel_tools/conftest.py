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

"""Fixtures for the OTel tool tests: the two trace populations from APP-6967."""

from typing import Any

import pytest

from tests.drmcp.unit.otel_tools.trace_factory import build_oversized_agent_trace
from tests.drmcp.unit.otel_tools.trace_factory import build_small_trace


@pytest.fixture
def oversized_agent_trace() -> dict[str, Any]:
    """Build the population this ticket exists for: 12 spans, 1,022,000 payload chars."""
    return build_oversized_agent_trace()


@pytest.fixture
def small_trace() -> dict[str, Any]:
    """Build the other population: a non-agentic trace of ~450 tokens end to end."""
    return build_small_trace()
