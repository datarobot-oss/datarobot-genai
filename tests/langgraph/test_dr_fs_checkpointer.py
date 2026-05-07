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

from copy import deepcopy

from fsspec.implementations.memory import MemoryFileSystem
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint

from datarobot_genai.langgraph.dr_fs_checkpointer import DataRobotFileSystemSaver


def test_data_robot_file_system_saver_roundtrip() -> None:
    fs = MemoryFileSystem()
    root = "/cp-root"
    saver = DataRobotFileSystemSaver(fs=fs, root=root)
    thread_id = "t1"
    checkpoint_ns = ""
    base_cfg: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns},
    }
    cp = empty_checkpoint()
    meta = {"source": "input", "step": -1}
    new_versions = deepcopy(cp["channel_versions"])
    out_cfg = saver.put(base_cfg, cp, meta, new_versions)
    tup = saver.get_tuple(out_cfg)
    assert tup is not None
    assert tup.checkpoint["id"] == cp["id"]
