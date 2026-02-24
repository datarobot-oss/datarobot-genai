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

import logging

from nat.front_ends.fastapi.fastapi_front_end_plugin import FastApiFrontEndPlugin
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import FastApiFrontEndPluginWorker
from nat.front_ends.fastapi.fastapi_front_end_plugin_worker import SessionManager
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.runtime.loader import WorkflowBuilder

from datarobot_genai.dragent.session import DRAgentAGUISessionManager
from datarobot_genai.dragent.step_adaptor import DRAgentNestedReasoningStepAdaptor

logger = logging.getLogger(__name__)


class DRAgentFastApiFrontEndPluginWorker(FastApiFrontEndPluginWorker):
    def get_step_adaptor(self) -> StepAdaptor:
        return DRAgentNestedReasoningStepAdaptor(self.front_end_config.step_adaptor)

    async def _create_session_manager(
        self, builder: WorkflowBuilder, entry_function: str | None = None
    ) -> SessionManager:
        """Create and register a SessionManager."""
        sm = await DRAgentAGUISessionManager.create(
            config=self._config, shared_builder=builder, entry_function=entry_function
        )
        self._session_managers.append(sm)

        return sm


class DRAgentFastApiFrontEndPlugin(FastApiFrontEndPlugin):
    def get_worker_class(self) -> type[FastApiFrontEndPluginWorker]:
        return DRAgentFastApiFrontEndPluginWorker
