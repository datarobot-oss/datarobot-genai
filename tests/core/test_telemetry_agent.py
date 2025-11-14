# Copyright 2025 DataRobot, Inc. and its affiliates.
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

from datarobot_genai.core.telemetry_agent import instrument


def test_instrument_idempotent_no_framework() -> None:
    instrument()
    instrument()  # idempotent


def test_instrument_with_frameworks() -> None:
    # Calls should not raise even if instrumentation packages are present or missing
    instrument("crewai")
    instrument("langgraph")
    instrument("llamaindex")
    # Repeat to ensure idempotency
    instrument("crewai")
    instrument("langgraph")
    instrument("llamaindex")
