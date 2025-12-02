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

from typing import Any

import pytest

import datarobot_genai.crewai.base as base_mod


@pytest.fixture
def mock_mcp_context(monkeypatch: Any) -> None:
    """Mock MCP tools context to return empty tools list."""

    class _Ctx:
        def __enter__(self) -> list[Any]:
            return []

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            pass

    def _ctx_factory(**_: Any) -> Any:
        return _Ctx()

    monkeypatch.setattr(base_mod, "mcp_tools_context", _ctx_factory, raising=True)
