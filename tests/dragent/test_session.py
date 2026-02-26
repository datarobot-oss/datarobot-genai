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

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from nat.data_models.config import Config
from nat.data_models.config import GeneralConfig
from nat.runtime.session import SessionManager

from datarobot_genai.dragent.request import DRAgentRunAgentInput
from datarobot_genai.dragent.response import DRAgentEventResponse
from datarobot_genai.dragent.session import DRAgentAGUISessionManager


@pytest.fixture
def session_manager():
    """Create a DRAgentAGUISessionManager with minimal config and mocked registry."""
    config = Config(general=GeneralConfig())
    shared_builder = MagicMock()
    shared_workflow = MagicMock()

    mock_registration = MagicMock()
    mock_registration.is_per_user = False

    with patch("nat.cli.type_registry.GlobalTypeRegistry.get") as get_registry:
        get_registry.return_value.get_function.return_value = mock_registration
        manager = DRAgentAGUISessionManager(
            config=config,
            shared_builder=shared_builder,
            shared_workflow=shared_workflow,
        )
        yield manager


class TestDRAgentAGUISessionManager:
    def test_is_session_manager_subclass(self):
        assert issubclass(DRAgentAGUISessionManager, SessionManager)

    def test_get_workflow_input_schema_returns_dragent_run_agent_input(self, session_manager):
        schema = session_manager.get_workflow_input_schema()
        assert schema is DRAgentRunAgentInput

    def test_get_workflow_streaming_output_schema_returns_dragent_event_response(
        self, session_manager
    ):
        schema = session_manager.get_workflow_streaming_output_schema()
        assert schema is DRAgentEventResponse
