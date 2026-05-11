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

import atexit
from copy import deepcopy

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint

from datarobot_genai.langgraph.dr_fs_checkpointer import DataRobotFileSystemSaver
from datarobot_genai.langgraph.dr_fs_checkpointer import _decoder_segment
from datarobot_genai.langgraph.dr_fs_checkpointer import _encoder_segment
from datarobot_genai.langgraph.dr_fs_checkpointer import _normalize_dr_fs_root
from datarobot_genai.langgraph.dr_fs_checkpointer import _path_join
from datarobot_genai.langgraph.dr_fs_checkpointer import _register_checkpoint_root_cleanup
from datarobot_genai.langgraph.dr_fs_checkpointer import _resolved_checkpoint_root


def test_normalize_dr_fs_root_preserves_dr_colon_slash_slash() -> None:
    assert _normalize_dr_fs_root("dr://") == "dr://"


def test_path_join_preserves_bare_dr_scheme() -> None:
    assert _path_join("dr://", "threads", "seg") == "dr://threads/seg"


def test_resolved_checkpoint_root_null_is_dr_scheme_root() -> None:
    assert _resolved_checkpoint_root(None) == "dr://"
    assert _resolved_checkpoint_root("") == "dr://"
    assert _resolved_checkpoint_root("   ") == "dr://"


def test_encoder_empty_string_is_non_empty_path_segment() -> None:
    assert _encoder_segment("") != ""
    assert _decoder_segment(_encoder_segment("")) == ""


def test_list_with_default_empty_checkpoint_ns_does_not_crash() -> None:
    fs = MemoryFileSystem()
    root = "/cp-root"
    saver = DataRobotFileSystemSaver(fs=fs, root=root)
    thread_id = "t-list"
    base_cfg: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""},
    }
    cp = empty_checkpoint()
    meta = {"source": "input", "step": -1}
    new_versions = deepcopy(cp["channel_versions"])
    out_cfg = saver.put(base_cfg, cp, meta, new_versions)
    items = list(saver.list(None))
    assert len(items) >= 1
    assert items[0].config["configurable"]["checkpoint_ns"] == ""
    assert items[0].config["configurable"]["thread_id"] == thread_id
    # Scoped list for same thread + empty ns
    scoped = list(saver.list({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}))
    assert len(scoped) >= 1
    assert saver.get_tuple(out_cfg) is not None


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


def test_register_checkpoint_root_cleanup_runs_once_and_removes_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import datarobot_genai.langgraph.dr_fs_checkpointer as dr_cp

    dr_cp._cleanup_registered_roots.clear()
    callbacks: list[object] = []

    def _capture_register(fn: object) -> None:
        callbacks.append(fn)

    monkeypatch.setattr(atexit, "register", _capture_register)

    fs = MemoryFileSystem()
    root = "/ephemeral/cp"
    with fs.open(f"{root}/marker.txt", "wb") as f:
        f.write(b"x")

    _register_checkpoint_root_cleanup(fs, root)
    _register_checkpoint_root_cleanup(fs, root)
    assert len(callbacks) == 1
    assert fs.exists(root)
    callbacks[0]()
    assert not fs.exists(root)


def test_register_checkpoint_root_cleanup_tracks_dr_scheme_not_dr_colon() -> None:
    import datarobot_genai.langgraph.dr_fs_checkpointer as dr_cp

    dr_cp._cleanup_registered_roots.clear()
    fs = MemoryFileSystem()
    _register_checkpoint_root_cleanup(fs, "dr://")
    assert dr_cp._cleanup_registered_roots == {"dr://"}


def test_reset_default_langgraph_checkpointer_clears_caches() -> None:
    import datarobot_genai.langgraph.dr_fs_checkpointer as dr_cp

    dr_cp._default_process_checkpointers["k"] = object()  # type: ignore[assignment]
    dr_cp._cleanup_registered_roots.add("/x")
    dr_cp.reset_default_langgraph_checkpointer_for_tests()
    assert dr_cp._default_process_checkpointers == {}
    assert dr_cp._cleanup_registered_roots == set()
