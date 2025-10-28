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

from __future__ import annotations

from datarobot_genai import agents


def test_agents_public_api_imports() -> None:
    assert hasattr(agents, "BaseAgent")
    assert hasattr(agents, "make_system_prompt")
    assert hasattr(agents, "extract_user_prompt_content")


def test_base_agent_litellm_api_base_smoke() -> None:
    a = agents.BaseAgent(api_base="https://app.datarobot.com/api/v2")
    assert a.litellm_api_base("dep-1").endswith("/deployments/dep-1/chat/completions")
