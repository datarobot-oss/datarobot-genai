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

"""Test fixtures for ResourceStore tests."""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from datarobot_genai.drmcp.core.resource_store.backends.filesystem import FilesystemBackend
from datarobot_genai.drmcp.core.resource_store.store import ResourceStore


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def backend(temp_dir: Path) -> FilesystemBackend:
    """Create a FilesystemBackend instance for testing."""
    return FilesystemBackend(temp_dir)


@pytest.fixture
def store(backend: FilesystemBackend) -> ResourceStore:
    """Create a ResourceStore instance for testing."""
    return ResourceStore(backend)
