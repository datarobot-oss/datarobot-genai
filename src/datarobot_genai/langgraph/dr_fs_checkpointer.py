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
"""LangGraph checkpoint persistence on a DataRobot (fsspec) file system.

`DataRobotFileSystem` does not support ``makedirs``; this module relies on creating objects
by writing files at nested paths (same pattern as normal DR FS uploads).
"""

from __future__ import annotations

import atexit
import pickle
import random
from base64 import urlsafe_b64decode
from base64 import urlsafe_b64encode
from collections.abc import AsyncIterator
from collections.abc import Iterator
from collections.abc import Sequence
from hashlib import sha256
from typing import Any

from fsspec import AbstractFileSystem
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import WRITES_IDX_MAP
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.base import ChannelVersions
from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.base import SerializerProtocol
from langgraph.checkpoint.base import get_checkpoint_id
from langgraph.checkpoint.base import get_checkpoint_metadata

_default_process_checkpointers: dict[str, Any] = {}
_cleanup_registered_roots: set[str] = set()


def _register_checkpoint_root_cleanup(fs: AbstractFileSystem, root: str) -> None:
    """Remove ``root`` recursively on interpreter exit (best-effort).

    Each distinct ``root`` registers at most one cleanup callback per process.
    """
    root_norm = root.rstrip("/")
    if root_norm in _cleanup_registered_roots:
        return
    _cleanup_registered_roots.add(root_norm)

    def _cleanup() -> None:
        try:
            if fs.exists(root_norm):
                fs.rm(root_norm, recursive=True)
        except Exception:
            pass

    atexit.register(_cleanup)


def _path_join(*parts: str) -> str:
    if not parts:
        return ""
    out = parts[0].rstrip("/")
    for p in parts[1:]:
        seg = p.strip("/")
        if seg:
            out = f"{out}/{seg}"
    return out


# ``urlsafe_b64encode(b"")`` decodes to an empty string; ``_path_join`` drops empty
# segments. That folded the default empty ``checkpoint_ns`` into ``.../threads/.../ns`` so
# ``cpts`` / ``blobs`` / ``writes`` sat directly under ``ns`` and ``list()`` tried to
# base64-decode those directory names as namespaces.
_EMPTY_SEGMENT_TOKEN = "dr_genai_langgraph_empty_segment"


def _encoder_segment(s: str) -> str:
    if s == "":
        return _EMPTY_SEGMENT_TOKEN
    return urlsafe_b64encode(s.encode()).decode().rstrip("=")


def _decoder_segment(enc: str) -> str:
    if enc == _EMPTY_SEGMENT_TOKEN:
        return ""
    pad = "=" * ((4 - len(enc) % 4) % 4)
    return urlsafe_b64decode(enc + pad).decode()


def _blob_filename(channel: str, version: str | int | float) -> str:
    key = f"{channel}\0{version!s}".encode()
    return sha256(key).hexdigest()


class DataRobotFileSystemSaver(BaseCheckpointSaver[str]):
    """Persist LangGraph checkpoints on an fsspec filesystem.

    For example :class:`datarobot.fs.DataRobotFileSystem`.

    Layout under ``root``:

    - ``threads/<b64(thread_id)>/ns/<b64(checkpoint_ns)>/cpts/<checkpoint_id>.pkl``
      (empty ``thread_id`` or ``checkpoint_ns`` uses ``dr_genai_langgraph_empty_segment``
      instead of an empty path segment)
    - ``threads/.../writes/<checkpoint_id>.pkl`` (pickled writes dict)
    - ``threads/.../blobs/<sha256(channel,version)>.pkl`` (pickled ``serde.dumps_typed`` result)

    ``root`` must be a path the filesystem accepts, for example ``dr://`` or
    ``dr://<catalog_id>/langgraph_checkpoints``.
    """

    def __init__(
        self,
        *,
        fs: AbstractFileSystem,
        root: str,
        serde: SerializerProtocol | None = None,
    ) -> None:
        super().__init__(serde=serde)
        self.fs = fs
        self.root = _normalize_dr_fs_root(root)

    def _ns_root(self, thread_id: str, checkpoint_ns: str) -> str:
        return _path_join(
            self.root,
            "threads",
            _encoder_segment(thread_id),
            "ns",
            _encoder_segment(checkpoint_ns),
        )

    def _blob_path(
        self,
        thread_id: str,
        checkpoint_ns: str,
        channel: str,
        version: str | int | float,
    ) -> str:
        return _path_join(
            self._ns_root(thread_id, checkpoint_ns),
            "blobs",
            f"{_blob_filename(channel, version)}.pkl",
        )

    def _cpt_path(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return _path_join(self._ns_root(thread_id, checkpoint_ns), "cpts", f"{checkpoint_id}.pkl")

    def _writes_path(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return _path_join(self._ns_root(thread_id, checkpoint_ns), "writes", f"{checkpoint_id}.pkl")

    def _load_writes_map(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]]:
        path = self._writes_path(thread_id, checkpoint_ns, checkpoint_id)
        if not self.fs.exists(path):
            return {}
        with self.fs.open(path, "rb") as f:
            return pickle.load(f)

    def _save_writes_map(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        data: dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]],
    ) -> None:
        path = self._writes_path(thread_id, checkpoint_ns, checkpoint_id)
        with self.fs.open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_blobs(
        self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> dict[str, Any]:
        channel_values: dict[str, Any] = {}
        for k, v in versions.items():
            bp = self._blob_path(thread_id, checkpoint_ns, k, v)
            if not self.fs.exists(bp):
                continue
            with self.fs.open(bp, "rb") as f:
                typed = pickle.load(f)
            if typed[0] != "empty":
                channel_values[k] = self.serde.loads_typed(typed)
        return channel_values

    def _tuple_from_saved(
        self,
        *,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
        saved: tuple[tuple[str, bytes], tuple[str, bytes], str | None],
        response_config: RunnableConfig,
    ) -> CheckpointTuple:
        checkpoint_bytes, metadata_bytes, parent_checkpoint_id = saved
        writes_map = self._load_writes_map(thread_id, checkpoint_ns, checkpoint_id)
        writes = writes_map.values()
        checkpoint_: Checkpoint = self.serde.loads_typed(checkpoint_bytes)
        return CheckpointTuple(
            config=response_config,
            checkpoint={
                **checkpoint_,
                "channel_values": self._load_blobs(
                    thread_id, checkpoint_ns, checkpoint_["channel_versions"]
                ),
            },
            metadata=self.serde.loads_typed(metadata_bytes),
            pending_writes=[(wid, c, self.serde.loads_typed(v)) for wid, c, v, _ in writes],
            parent_config=(
                {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
                if parent_checkpoint_id
                else None
            ),
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id: str = config["configurable"]["thread_id"]
        checkpoint_ns: str = config["configurable"].get("checkpoint_ns", "")
        ns_root = self._ns_root(thread_id, checkpoint_ns)
        cpts_dir = _path_join(ns_root, "cpts")
        if checkpoint_id := get_checkpoint_id(config):
            path = self._cpt_path(thread_id, checkpoint_ns, checkpoint_id)
            if not self.fs.exists(path):
                return None
            with self.fs.open(path, "rb") as f:
                saved = pickle.load(f)
            return self._tuple_from_saved(
                thread_id=thread_id,
                checkpoint_ns=checkpoint_ns,
                checkpoint_id=checkpoint_id,
                saved=saved,
                response_config=config,
            )
        if not self.fs.exists(cpts_dir):
            return None
        entries = self.fs.ls(cpts_dir, detail=False)
        if not entries:
            return None
        ids: list[str] = []
        for e in entries:
            name = e.rsplit("/", 1)[-1]
            if name.endswith(".pkl"):
                ids.append(name[: -len(".pkl")])
        if not ids:
            return None
        checkpoint_id = max(ids)
        path = self._cpt_path(thread_id, checkpoint_ns, checkpoint_id)
        with self.fs.open(path, "rb") as f:
            saved = pickle.load(f)
        return self._tuple_from_saved(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_id=checkpoint_id,
            saved=saved,
            response_config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            },
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        threads_base = _path_join(self.root, "threads")
        if config:
            thread_ids = [config["configurable"]["thread_id"]]
        else:
            if not self.fs.exists(threads_base):
                return
            thread_ids = []
            for e in self.fs.ls(threads_base, detail=False):
                seg = e.rstrip("/").rsplit("/", 1)[-1]
                thread_ids.append(_decoder_segment(seg))

        config_checkpoint_ns = config["configurable"].get("checkpoint_ns") if config else None
        config_checkpoint_id = get_checkpoint_id(config) if config else None

        for thread_id in thread_ids:
            troot = _path_join(threads_base, _encoder_segment(thread_id), "ns")
            if not self.fs.exists(troot):
                continue
            for ns_entry in self.fs.ls(troot, detail=False):
                enc_ns = ns_entry.rstrip("/").rsplit("/", 1)[-1]
                checkpoint_ns = _decoder_segment(enc_ns)
                if config_checkpoint_ns is not None and checkpoint_ns != config_checkpoint_ns:
                    continue
                cpts_dir = _path_join(ns_entry.rstrip("/"), "cpts")
                if not self.fs.exists(cpts_dir):
                    continue
                checkpoint_entries: list[tuple[str, str]] = []
                for cp_entry in self.fs.ls(cpts_dir, detail=False):
                    fname = cp_entry.rsplit("/", 1)[-1]
                    if not fname.endswith(".pkl"):
                        continue
                    cid = fname[: -len(".pkl")]
                    checkpoint_entries.append((cid, cp_entry))
                for checkpoint_id, path in sorted(
                    checkpoint_entries,
                    key=lambda x: x[0],
                    reverse=True,
                ):
                    if config_checkpoint_id and checkpoint_id != config_checkpoint_id:
                        continue
                    if (
                        before
                        and (before_checkpoint_id := get_checkpoint_id(before))
                        and checkpoint_id >= before_checkpoint_id
                    ):
                        continue
                    with self.fs.open(path, "rb") as f:
                        saved = pickle.load(f)
                    metadata = self.serde.loads_typed(saved[1])
                    if filter and not all(
                        query_value == metadata.get(query_key)
                        for query_key, query_value in filter.items()
                    ):
                        continue
                    if limit is not None and limit <= 0:
                        return
                    elif limit is not None:
                        limit -= 1
                    yield self._tuple_from_saved(
                        thread_id=thread_id,
                        checkpoint_ns=checkpoint_ns,
                        checkpoint_id=checkpoint_id,
                        saved=saved,
                        response_config={
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": checkpoint_ns,
                                "checkpoint_id": checkpoint_id,
                            }
                        },
                    )

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        c = checkpoint.copy()
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        values: dict[str, Any] = c.pop("channel_values")  # type: ignore[misc]
        for k, v in new_versions.items():
            blob = self.serde.dumps_typed(values[k]) if k in values else ("empty", b"")
            bp = self._blob_path(thread_id, checkpoint_ns, k, v)
            with self.fs.open(bp, "wb") as f:
                pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        saved = (
            self.serde.dumps_typed(c),
            self.serde.dumps_typed(get_checkpoint_metadata(config, metadata)),
            config["configurable"].get("checkpoint_id"),
        )
        cpp = self._cpt_path(thread_id, checkpoint_ns, checkpoint["id"])
        with self.fs.open(cpp, "wb") as f:
            pickle.dump(saved, f, protocol=pickle.HIGHEST_PROTOCOL)
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint["id"],
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        writes_map = self._load_writes_map(thread_id, checkpoint_ns, checkpoint_id)
        outer_writes_ = writes_map or None
        for idx, (c, v) in enumerate(writes):
            inner_key = (task_id, WRITES_IDX_MAP.get(c, idx))
            if inner_key[1] >= 0 and outer_writes_ and inner_key in outer_writes_:
                continue
            writes_map[inner_key] = (
                task_id,
                c,
                self.serde.dumps_typed(v),
                task_path,
            )
        self._save_writes_map(thread_id, checkpoint_ns, checkpoint_id, writes_map)

    def delete_thread(self, thread_id: str) -> None:
        tdir = _path_join(self.root, "threads", _encoder_segment(thread_id))
        if self.fs.exists(tdir):
            self.fs.rm(tdir, recursive=True)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        return self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        return self.delete_thread(thread_id)

    def get_next_version(self, current: str | None, channel: None) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, int):
            current_v = current
        else:
            current_v = int(current.split(".")[0])
        next_v = current_v + 1
        next_h = random.random()
        return f"{next_v:032}.{next_h:016}"


def _normalize_checkpoint_base(checkpoint_base: str | None) -> str | None:
    if checkpoint_base is None:
        return None
    s = checkpoint_base.strip()
    return s or None


def _normalize_dr_fs_root(root: str) -> str:
    """Normalize filesystem root; preserve bare ``dr://`` (plain rstrip yields ``dr:``)."""
    trimmed = root.rstrip("/")
    return "dr://" if trimmed == "dr:" else trimmed


def _resolved_checkpoint_root(checkpoint_base: str | None) -> str:
    """Resolve saver root; null/empty ``checkpoint_base`` uses ``dr://``."""
    normalized = _normalize_checkpoint_base(checkpoint_base)
    if not normalized:
        return "dr://"
    return _normalize_dr_fs_root(normalized)


def default_langgraph_checkpointer(
    *, checkpoint_base: str | None = None
) -> DataRobotFileSystemSaver:
    """Return a process-wide default saver using :class:`datarobot.fs.DataRobotFileSystem`.

    ``checkpoint_base`` is the optional ``dr://`` prefix for checkpoint files (typically passed
    from application configuration). If unset or empty, the root is ``dr://``. Best-effort removal
    of each distinct root is registered with :mod:`atexit`. For checkpoints that must survive
    process shutdown, construct and pass your own ``checkpointer=`` instead of relying on this
    default.
    """
    root = _resolved_checkpoint_root(checkpoint_base)

    if (existing := _default_process_checkpointers.get(root)) is not None:
        return existing
    from datarobot.fs import DataRobotFileSystem

    fs = DataRobotFileSystem()
    saver = DataRobotFileSystemSaver(fs=fs, root=root)
    _default_process_checkpointers[saver.root] = saver
    _register_checkpoint_root_cleanup(fs, saver.root)
    return saver


def reset_default_langgraph_checkpointer_for_tests() -> None:
    """Clear process-wide defaults (for test isolation)."""
    _default_process_checkpointers.clear()
    _cleanup_registered_roots.clear()
