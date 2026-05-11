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

"""LangGraph checkpoint conformance against :class:`DataRobotFileSystemSaver`."""

from __future__ import annotations

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from langgraph.checkpoint.conformance import checkpointer_test
from langgraph.checkpoint.conformance import validate

from datarobot_genai.langgraph.dr_fs_checkpointer import DataRobotFileSystemSaver


@checkpointer_test(name="DataRobotFileSystemSaver")
async def datarobot_fs_memory_checkpointer():
    # GIVEN an isolated in-memory fsspec tree for this capability suite
    fs = MemoryFileSystem()
    root = "/langgraph_checkpoint_conformance"
    # WHEN the saver is constructed
    yield DataRobotFileSystemSaver(fs=fs, root=root)
    # THEN validate() exercises put/get/list/delete via async APIs


@pytest.mark.asyncio
async def test_datarobot_fs_checkpointer_langgraph_conformance() -> None:
    report = await validate(datarobot_fs_memory_checkpointer, progress=None)
    report.print_report()
    assert report.passed_all_base()
