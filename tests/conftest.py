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

from typing import Any

import pytest

from datarobot_genai.core.mcp.common import MCPConfig


def make_run_agent_input_with_history() -> Any:
    """Create a RunAgentInput with multi-turn chat history for agent tests.

    Use this for tests that verify chat history functionality across
    drmcp tools tests don't use ag-ui-protocol dependency.
    """
    from ag_ui.core import AssistantMessage
    from ag_ui.core import RunAgentInput
    from ag_ui.core import SystemMessage as AgSystemMessage
    from ag_ui.core import UserMessage

    return RunAgentInput(
        messages=[
            AgSystemMessage(id="sys_1", content="You are a helper."),
            UserMessage(id="user_1", content="First question"),
            AssistantMessage(id="asst_1", content="First answer"),
            UserMessage(id="user_2", content="Follow-up"),
        ],
        tools=[],
        forwarded_props=dict(model="m", authorization_context={}, forwarded_headers={}),
        thread_id="thread_id",
        run_id="run_id",
        state={},
        context=[],
    )


@pytest.fixture
def run_agent_input_with_history() -> Any:
    """Fixture providing a RunAgentInput with multi-turn chat history."""
    return make_run_agent_input_with_history()


@pytest.fixture
def agent_auth_context_data() -> dict[str, Any]:
    """Return sample authorization context data with required AuthCtx fields."""
    return {
        "user": {"id": "123", "name": "foo", "email": "foo@example.com"},
        "identities": [
            {"id": "id123", "type": "user", "provider_type": "github", "provider_user_id": "123"}
        ],
    }


@pytest.fixture(autouse=True, scope="function")
def disable_env_file(monkeypatch):
    """Disable loading of .env file for MCPConfig and related settings classes.

    Pydantic BaseSettings uses ``model_config.env_file`` to pull values from an env file.
    Tests should rely solely on explicit parameters / injected environment variables.
    This fixture patches the class-level config so any implicit .env lookup is skipped.
    """
    # Pydantic v2: model_config is a mapping; ensure env_file fields are disabled.
    if hasattr(MCPConfig, "model_config") and isinstance(getattr(MCPConfig, "model_config"), dict):
        # Create a shallow copy to avoid mutating original dict in-place across tests.
        new_config = {**MCPConfig.model_config}
        new_config["env_file"] = None
        new_config["env_file_encoding"] = None
        monkeypatch.setattr(MCPConfig, "model_config", new_config, raising=False)
    else:
        # Fallback: attempt attribute patching if object-like.
        monkeypatch.setattr(MCPConfig.model_config, "env_file", None, raising=False)
        monkeypatch.setattr(MCPConfig.model_config, "env_file_encoding", None, raising=False)
