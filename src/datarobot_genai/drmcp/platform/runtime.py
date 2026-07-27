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

"""MSF service runtime for drmcp."""

from __future__ import annotations

from msf.databases.mongodb.connection import AsyncMongodbConnectionProvider
from msf.runtime import AsyncProviderT
from msf.runtime import ServiceRuntime
from msf.runtime import Singleton

from datarobot_genai.drmcp.core.tool_sets.tool_sets_repo import ToolSetRepository
from datarobot_genai.drmcp.platform.mongo_config import DRMcpServiceConfig

_runtime: Runtime | None = None


class Runtime(ServiceRuntime):
    """Runtime wiring for MongoDB repositories."""

    config: Singleton[DRMcpServiceConfig] = Singleton(DRMcpServiceConfig)
    database = AsyncMongodbConnectionProvider(config=config.provided.mongo)

    tool_set_repository: AsyncProviderT[ToolSetRepository] = Singleton(
        ToolSetRepository.new,
        db=database,
    )


def get_runtime() -> Runtime:
    """Return the module-level ``Runtime`` singleton."""
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime


def reset_runtime() -> None:
    """Reset the module-level runtime (for tests and shutdown)."""
    global _runtime
    _runtime = None


async def close_runtime() -> None:
    """Close the Motor client and reset the runtime.

    Notebooks MSF services do not define a shared runtime shutdown hook; closing the
    underlying Motor client is the standard cleanup for async Mongo access.
    """
    global _runtime
    if _runtime is None:
        return

    try:
        db = await _runtime.database()
        db.client.close()
    finally:
        _runtime = None
