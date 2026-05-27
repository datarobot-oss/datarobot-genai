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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from copy import deepcopy

import pytest
from fsspec.implementations.memory import MemoryFileSystem
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import WRITES_IDX_MAP
from langgraph.checkpoint.base import empty_checkpoint

from datarobot_genai.langgraph.dr_fs_checkpointer import DataRobotFileSystemCheckpointSaver
from datarobot_genai.langgraph.dr_fs_checkpointer import _decoder_segment
from datarobot_genai.langgraph.dr_fs_checkpointer import _encoder_segment
from datarobot_genai.langgraph.dr_fs_checkpointer import _normalize_dr_fs_root
from datarobot_genai.langgraph.dr_fs_checkpointer import _pack_blob_file
from datarobot_genai.langgraph.dr_fs_checkpointer import _pack_cpt_file
from datarobot_genai.langgraph.dr_fs_checkpointer import _pack_typed_frame
from datarobot_genai.langgraph.dr_fs_checkpointer import _pack_writes_map_file
from datarobot_genai.langgraph.dr_fs_checkpointer import _path_join
from datarobot_genai.langgraph.dr_fs_checkpointer import _register_checkpoint_root_cleanup
from datarobot_genai.langgraph.dr_fs_checkpointer import _resolved_checkpoint_root
from datarobot_genai.langgraph.dr_fs_checkpointer import _unpack_blob_file
from datarobot_genai.langgraph.dr_fs_checkpointer import _unpack_cpt_file
from datarobot_genai.langgraph.dr_fs_checkpointer import _unpack_typed_frame
from datarobot_genai.langgraph.dr_fs_checkpointer import _unpack_writes_map_file


def test_pack_unpack_typed_frame_roundtrip() -> None:
    typed = ("msgpack", b"\x00\xffhello")
    raw = _pack_typed_frame(typed)
    out, end = _unpack_typed_frame(raw, 0)
    assert out == typed
    assert end == len(raw)


def test_writes_map_file_roundtrip() -> None:
    data: dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]] = {
        ("tid", 0): ("tid", "__interrupt__", ("json", b"{}"), ""),
    }
    raw = _pack_writes_map_file(data)
    assert _unpack_writes_map_file(raw) == data


def test_cpt_file_roundtrip() -> None:
    saved = (("json", b"cp"), ("json", b"meta"), "parent-1")
    raw = _pack_cpt_file(saved)
    assert _unpack_cpt_file(raw) == saved


def test_blob_file_roundtrip() -> None:
    typed = ("empty", b"")
    raw = _pack_blob_file(typed)
    assert _unpack_blob_file(raw) == typed


def test_normalize_dr_fs_root_preserves_dr_colon_slash_slash() -> None:
    assert _normalize_dr_fs_root("dr://") == "dr://"


def test_path_join_preserves_bare_dr_scheme() -> None:
    assert _path_join("dr://", "checkpoints", "seg") == "dr://checkpoints/seg"


def test_resolved_checkpoint_root_null_is_dr_scheme_root() -> None:
    assert _resolved_checkpoint_root(None) == "dr://"
    assert _resolved_checkpoint_root("") == "dr://"
    assert _resolved_checkpoint_root("   ") == "dr://"


def test_get_next_version_single_dot_and_fixed_width_suffix() -> None:
    fs = MemoryFileSystem()
    saver = DataRobotFileSystemCheckpointSaver(fs=fs, root="/v")
    for current in (
        None,
        0,
        1,
        2.9,
        "00000000000000000000000000000005.0000000000000999",
    ):
        v = saver.get_next_version(current, None)
        assert v.count(".") == 1, v
        left, right = v.split(".", 1)
        assert len(left) == 32 and left.isdigit(), v
        assert len(right) == 16 and right.isdigit(), v


def test_encoder_empty_string_is_non_empty_path_segment() -> None:
    assert _encoder_segment("") != ""
    assert _decoder_segment(_encoder_segment("")) == ""


def test_list_with_default_empty_checkpoint_ns_does_not_crash() -> None:
    fs = MemoryFileSystem()
    root = "/cp-root"
    saver = DataRobotFileSystemCheckpointSaver(fs=fs, root=root)
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
    saver = DataRobotFileSystemCheckpointSaver(fs=fs, root=root)
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


def test_put_writes_concurrent_different_tasks_preserves_all_writes() -> None:
    """Parallel ``put_writes`` for the same checkpoint used to race on one map file."""
    fs = MemoryFileSystem()
    root = "/cp-root"
    saver = DataRobotFileSystemCheckpointSaver(fs=fs, root=root)
    thread_id = "t-par"
    checkpoint_ns = "ns"
    base_cfg: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns},
    }
    cp = empty_checkpoint()
    meta = {"source": "input", "step": -1}
    new_versions = deepcopy(cp["channel_versions"])
    out_cfg = saver.put(base_cfg, cp, meta, new_versions)
    checkpoint_id = out_cfg["configurable"]["checkpoint_id"]
    write_cfg: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }

    def _put(task_id: str, channel: str, value: str) -> None:
        saver.put_writes(write_cfg, [(channel, value)], task_id)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(_put, f"task-{i}", f"channel-{i}", f"value-{i}") for i in range(16)]
        wait(futures)
        for f in futures:
            f.result()

    tup = saver.get_tuple(write_cfg)
    assert tup is not None and tup.pending_writes is not None
    channels = {pw[1] for pw in tup.pending_writes}
    assert channels == {f"channel-{i}" for i in range(16)}


def test_put_writes_skips_duplicate_inner_key_same_batch_when_writes_map_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: empty on-disk writes map must not disable the WRITES_IDX_MAP duplicate guard."""
    dup_channel = "__dr_genai_test_dup_write__"
    monkeypatch.setattr(
        "datarobot_genai.langgraph.dr_fs_checkpointer.WRITES_IDX_MAP",
        {**WRITES_IDX_MAP, dup_channel: 0},
    )

    fs = MemoryFileSystem()
    root = "/cp-root"
    saver = DataRobotFileSystemCheckpointSaver(fs=fs, root=root)
    thread_id = "t-dup"
    checkpoint_ns = "ns"
    base_cfg: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns},
    }
    cp = empty_checkpoint()
    meta = {"source": "input", "step": -1}
    new_versions = deepcopy(cp["channel_versions"])
    out_cfg = saver.put(base_cfg, cp, meta, new_versions)
    write_cfg: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": out_cfg["configurable"]["checkpoint_id"],
        }
    }
    task_id = "task-1"
    saver.put_writes(
        write_cfg,
        [(dup_channel, "first"), (dup_channel, "second")],
        task_id,
    )
    tup = saver.get_tuple(write_cfg)
    assert tup is not None
    assert tup.pending_writes is not None
    assert len(tup.pending_writes) == 1
    assert tup.pending_writes[0][2] == "first"


def test_register_checkpoint_root_cleanup_runs_once_and_removes_checkpoints_only(
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
    checkpoints = f"{root}/checkpoints"
    with fs.open(f"{root}/marker.txt", "wb") as f:
        f.write(b"x")
    with fs.open(f"{checkpoints}/leaf.bin", "wb") as f:
        f.write(b"y")

    _register_checkpoint_root_cleanup(fs, root)
    _register_checkpoint_root_cleanup(fs, root)
    assert len(callbacks) == 1
    assert fs.exists(root)
    assert fs.exists(checkpoints)
    callbacks[0]()
    assert fs.exists(f"{root}/marker.txt")
    assert not fs.exists(checkpoints)


def test_register_checkpoint_root_cleanup_tracks_dr_scheme_not_dr_colon() -> None:
    import datarobot_genai.langgraph.dr_fs_checkpointer as dr_cp

    dr_cp._cleanup_registered_roots.clear()
    fs = MemoryFileSystem()
    _register_checkpoint_root_cleanup(fs, "dr://")
    assert dr_cp._cleanup_registered_roots == {"dr://checkpoints"}


def test_reset_default_langgraph_checkpointer_clears_caches() -> None:
    import datarobot_genai.langgraph.dr_fs_checkpointer as dr_cp

    dr_cp._default_process_checkpointers["k"] = object()  # type: ignore[assignment]
    dr_cp._cleanup_registered_roots.add("/x")
    dr_cp.reset_default_langgraph_checkpointer_for_tests()
    assert dr_cp._default_process_checkpointers == {}
    assert dr_cp._cleanup_registered_roots == set()
