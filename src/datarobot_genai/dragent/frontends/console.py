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

import logging

from nat.front_ends.console.console_front_end_config import ConsoleFrontEndConfig
from nat.front_ends.console.console_front_end_plugin import ConsoleFrontEndPlugin

logger = logging.getLogger(__name__)


# Extends NAT's ConsoleFrontEndConfig with a distinct name so it can be registered
# as a separate frontend type for `nat dragent run`.
# Base: https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/src/nat/front_ends/console/console_front_end_config.py
class DRAgentConsoleFrontEndConfig(ConsoleFrontEndConfig, name="dragent_console"):  # type: ignore
    """Frontend config for running a dragent workflow from the console."""


# Extends NAT's ConsoleFrontEndPlugin. Currently identical to the base — the dragent-specific
# overrides (DRAgentAGUISessionManager, DRAgentNestedReasoningStepAdaptor, auth context injection)
# will be added in follow-up work.
# Base: https://github.com/NVIDIA/NeMo-Agent-Toolkit/blob/release/1.4/src/nat/front_ends/console/console_front_end_plugin.py
class DRAgentConsoleFrontEndPlugin(ConsoleFrontEndPlugin):
    """Console frontend for dragent workflows with AG-UI support."""
