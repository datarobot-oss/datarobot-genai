# Copyright 2025 DataRobot, Inc.
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
from collections.abc import Generator
from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from datarobot_genai.drmcp.core.feature_flags import FeatureFlag
from datarobot_genai.drmcp.core.lineage.manager import LineageManager


@pytest.fixture(autouse=True)
def mock_datarobot_token() -> Generator[None, None, None]:
    """Fixture to provide mock DataRobot API token.

    This fixture is automatically used in all unit tests to ensure
    DataRobot credentials validation passes.
    """
    with patch.dict(
        "os.environ",
        {
            "DATAROBOT_API_TOKEN": "test-token",
            "DATAROBOT_ENDPOINT": "https://app.datarobot.com/api/v2",
        },
    ):
        yield


@pytest.fixture(autouse=True)
def mock_all_telemetry(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Mock all telemetry-related functionality for unit tests.

    Skips mocking for test_shared_telemetry.py since those tests specifically test
    telemetry functionality.
    """
    # Skip for test_shared_telemetry.py tests
    if request.module.__name__.endswith("test_shared_telemetry"):
        yield
        return

    with (
        patch("datarobot_genai.drmcp.core.telemetry.initialize_telemetry", return_value=None),
        patch("opentelemetry.trace.get_tracer"),
        patch("opentelemetry.trace.get_tracer_provider"),
        patch("datarobot_genai.drmcp.core.telemetry._setup_otel_logging", return_value=None),
        patch("opentelemetry.exporter.otlp.proto.http._log_exporter.OTLPLogExporter"),
        patch.dict(
            "os.environ",
            {
                "OTEL_ENABLED": "false",
                "OTEL_EXPORTER_OTLP_ENDPOINT": "",
                "OTEL_EXPORTER_OTLP_HEADERS": "",
            },
            clear=False,
        ),
    ):
        yield


@pytest.fixture
def mock_feature_flag_create() -> Iterator[Mock]:
    with patch.object(FeatureFlag, "create") as mock_func:
        yield mock_func


@pytest.fixture
def mock_lineage_manager_init() -> Iterator[Mock]:
    with patch.object(LineageManager, "__init__") as mock_func:
        mock_func.return_value = None
        yield mock_func


@pytest.fixture
def mock_sync_mcp_tools() -> Iterator[AsyncMock]:
    with patch.object(
        LineageManager,
        "sync_mcp_tools",
        new_callable=AsyncMock,
    ) as mock_func:
        yield mock_func


@pytest.fixture
def mock_sync_mcp_prompts() -> Iterator[AsyncMock]:
    with patch.object(
        LineageManager,
        "sync_mcp_prompts",
        new_callable=AsyncMock,
    ) as mock_func:
        yield mock_func
