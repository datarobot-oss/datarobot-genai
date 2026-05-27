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

When the saver ``root`` is the bare scheme ``dr://`` (the default from
:func:`default_langgraph_checkpointer` when ``checkpoint_base`` is unset), the filesystem
resolves the active catalog and stores artifacts under
``<catalog_id>/langgraph_checkpoints/checkpoints/`` (see
:class:`DataRobotFileSystemCheckpointSaver`).
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import random
import struct
import threading
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
from langgraph.checkpoint.base import ChannelProtocol
from langgraph.checkpoint.base import ChannelVersions
from langgraph.checkpoint.base import Checkpoint
from langgraph.checkpoint.base import CheckpointMetadata
from langgraph.checkpoint.base import CheckpointTuple
from langgraph.checkpoint.base import SerializerProtocol
from langgraph.checkpoint.base import get_checkpoint_id
from langgraph.checkpoint.base import get_checkpoint_metadata

_default_process_checkpointers: dict[str, Any] = {}
_cleanup_registered_roots: set[str] = set()
# Serialize load→merge→save for a single task shard (same ``task_id``, same checkpoint).
_task_writes_shard_locks: dict[str, threading.Lock] = {}
_task_writes_shard_locks_mutex = threading.Lock()


@contextlib.contextmanager
def _locked_task_writes_shard(shard_path: str) -> Iterator[None]:
    with _task_writes_shard_locks_mutex:
        lock = _task_writes_shard_locks.get(shard_path)
        if lock is None:
            lock = threading.Lock()
            _task_writes_shard_locks[shard_path] = lock
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


# On-disk subtree for all saver paths under ``root`` (not LangGraph ``thread_id``).
_CHECKPOINT_TREE_DIR = "checkpoints"
# Length-prefixed binary (``struct``) only — no file-type magic bytes. LangGraph ``serde`` payloads
# stay raw bytes inside typed frames; no pickle on disk. Files use ``.bin`` (not ``.pkl``).
_CHECKPOINT_ARTIFACT_EXT = ".bin"


def _pack_lenprefixed_utf8(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("!I", len(raw)) + raw


def _unpack_lenprefixed_utf8(data: bytes, offset: int) -> tuple[str, int]:
    (ln,) = struct.unpack_from("!I", data, offset)
    offset += 4
    end = offset + ln
    return data[offset:end].decode("utf-8"), end


def _pack_typed_frame(typed: tuple[str, bytes]) -> bytes:
    kind_b = typed[0].encode("utf-8")
    payload = typed[1]
    return struct.pack("!I", len(kind_b)) + kind_b + struct.pack("!I", len(payload)) + payload


def _unpack_typed_frame(data: bytes, offset: int) -> tuple[tuple[str, bytes], int]:
    (kln,) = struct.unpack_from("!I", data, offset)
    offset += 4
    kind = data[offset : offset + kln].decode("utf-8")
    offset += kln
    (dln,) = struct.unpack_from("!I", data, offset)
    offset += 4
    end = offset + dln
    return (kind, data[offset:end]), end


def _pack_optional_parent(parent: str | None) -> bytes:
    if parent is None:
        return b"\x00"
    raw = parent.encode("utf-8")
    return b"\x01" + struct.pack("!I", len(raw)) + raw


def _unpack_optional_parent(data: bytes, offset: int) -> tuple[str | None, int]:
    flag = data[offset]
    offset += 1
    if flag == 0:
        return None, offset
    (ln,) = struct.unpack_from("!I", data, offset)
    offset += 4
    end = offset + ln
    return data[offset:end].decode("utf-8"), end


def _pack_blob_file(typed: tuple[str, bytes]) -> bytes:
    """Pack one ``serde.dumps_typed`` tuple as length-prefixed binary.

    Layout: ``u32 len(kind) | kind utf-8 | u32 len(payload) | payload``.
    """
    return _pack_typed_frame(typed)


def _unpack_blob_file(data: bytes) -> tuple[str, bytes]:
    typed, end = _unpack_typed_frame(data, 0)
    if end != len(data):
        msg = "blob artifact length mismatch"
        raise ValueError(msg)
    return typed


def _pack_cpt_file(
    saved: tuple[tuple[str, bytes], tuple[str, bytes], str | None],
) -> bytes:
    """Checkpoint + metadata typed frames, then optional parent id (same encoding as before)."""
    c, m, parent = saved
    return _pack_typed_frame(c) + _pack_typed_frame(m) + _pack_optional_parent(parent)


def _unpack_cpt_file(data: bytes) -> tuple[tuple[str, bytes], tuple[str, bytes], str | None]:
    c, off = _unpack_typed_frame(data, 0)
    m, off = _unpack_typed_frame(data, off)
    parent, off = _unpack_optional_parent(data, off)
    if off != len(data):
        msg = "checkpoint artifact length mismatch"
        raise ValueError(msg)
    return (c, m, parent)


def _pack_writes_map_file(
    data: dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]],
) -> bytes:
    """``u32`` entry count, then repeated write records (see unpack)."""
    items = list(data.items())
    out = bytearray(struct.pack("!I", len(items)))
    for (task_id, write_idx), (_tid, channel, typed_v, task_path) in items:
        out += (
            _pack_lenprefixed_utf8(task_id)
            + struct.pack("!q", write_idx)
            + _pack_lenprefixed_utf8(channel)
            + _pack_typed_frame(typed_v)
            + _pack_lenprefixed_utf8(task_path)
        )
    return bytes(out)


def _unpack_writes_map_file(
    data: bytes,
) -> dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]]:
    off = 0
    (n,) = struct.unpack_from("!I", data, off)
    off += 4
    out: dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]] = {}
    for _ in range(n):
        task_id, off = _unpack_lenprefixed_utf8(data, off)
        (write_idx,) = struct.unpack_from("!q", data, off)
        off += 8
        channel, off = _unpack_lenprefixed_utf8(data, off)
        typed_v, off = _unpack_typed_frame(data, off)
        task_path, off = _unpack_lenprefixed_utf8(data, off)
        out[(task_id, int(write_idx))] = (task_id, channel, typed_v, task_path)
    if off != len(data):
        msg = "writes map artifact length mismatch"
        raise ValueError(msg)
    return out


def _normalize_dr_fs_root(root: str) -> str:
    """Normalize filesystem root; preserve bare ``dr://`` (plain rstrip yields ``dr:``)."""
    trimmed = root.rstrip("/")
    return "dr://" if trimmed == "dr:" else trimmed


def _register_checkpoint_root_cleanup(fs: AbstractFileSystem, root: str) -> None:
    """Remove ``<root>/checkpoints`` recursively on interpreter exit (best-effort).

    Checkpoints are stored only under the ``checkpoints/`` subtree (see
    :class:`DataRobotFileSystemCheckpointSaver`); we never delete the entire configured ``root``
    or the ``langgraph_checkpoints`` prefix (for example when ``root`` is ``dr://`` and the
    effective path is ``<catalog_id>/langgraph_checkpoints``) so other DR FS objects are preserved.

    Each distinct cleanup path registers at most one callback per process.
    """
    root_norm = _normalize_dr_fs_root(root)
    cleanup_path = _path_join(root_norm, _CHECKPOINT_TREE_DIR)
    if cleanup_path in _cleanup_registered_roots:
        return
    _cleanup_registered_roots.add(cleanup_path)

    def _cleanup() -> None:
        try:
            if fs.exists(cleanup_path):
                fs.rm(cleanup_path, recursive=True)
        except Exception:
            pass

    atexit.register(_cleanup)


def _path_join(*parts: str) -> str:
    if not parts:
        return ""
    out = _normalize_dr_fs_root(parts[0])
    for p in parts[1:]:
        seg = p.strip("/")
        if seg:
            # Bare ``dr://`` already ends with ``/``; avoid ``dr:///checkpoints`` (triple slash).
            if out.endswith("/"):
                out = f"{out}{seg}"
            else:
                out = f"{out}/{seg}"
    return out


# ``urlsafe_b64encode(b"")`` decodes to an empty string; ``_path_join`` drops empty
# segments. That folded the default empty ``checkpoint_ns`` into ``.../checkpoints/.../ns`` so
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


class DataRobotFileSystemCheckpointSaver(BaseCheckpointSaver[str]):
    """Persist LangGraph checkpoints on an fsspec filesystem.

    For example :class:`datarobot.fs.DataRobotFileSystem`.

    **Root and effective prefix**

    - ``root="dr://"`` (default via :func:`default_langgraph_checkpointer` when no
      ``checkpoint_base`` is set): :class:`datarobot.fs.DataRobotFileSystem` resolves the
      active catalog; objects are stored under
      ``<catalog_id>/langgraph_checkpoints/checkpoints/...``.
    - Explicit prefix, for example ``dr://<catalog_id>/langgraph_checkpoints``: same
      ``checkpoints/`` subtree is appended under that path.

    **Layout** (paths below are relative to the effective prefix ending in
    ``.../checkpoints/``):

    - ``<b64(thread_id)>/ns/<b64(checkpoint_ns)>/cpts/<checkpoint_id>.bin``
      (empty ``thread_id`` or ``checkpoint_ns`` uses ``dr_genai_langgraph_empty_segment``
      instead of an empty path segment)
    - ``.../writes/<checkpoint_id>/w_<sha256(task_id)>.bin`` (per-task writes shard;
      avoids lost updates when parallel branches call ``put_writes`` concurrently)
    - legacy ``.../writes/<checkpoint_id>.bin`` (monolithic map) is still read and merged
      when present
    - ``.../blobs/<sha256(channel,version)>.bin`` (single ``serde.dumps_typed`` frame)

    On-disk format is concatenated length-prefixed ``struct`` segments (no magic header, no
    pickle). Layout is implied by path (``blobs/`` vs ``cpts/`` vs ``writes/``).
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
            _CHECKPOINT_TREE_DIR,
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
            f"{_blob_filename(channel, version)}{_CHECKPOINT_ARTIFACT_EXT}",
        )

    def _cpt_path(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return _path_join(
            self._ns_root(thread_id, checkpoint_ns),
            "cpts",
            f"{checkpoint_id}{_CHECKPOINT_ARTIFACT_EXT}",
        )

    def _writes_legacy_path(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        """Pre-shard monolithic writes map (still merged on read)."""
        return _path_join(
            self._ns_root(thread_id, checkpoint_ns),
            "writes",
            f"{checkpoint_id}{_CHECKPOINT_ARTIFACT_EXT}",
        )

    def _writes_shard_dir(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return _path_join(
            self._ns_root(thread_id, checkpoint_ns),
            "writes",
            checkpoint_id,
        )

    def _task_writes_shard_path(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str, task_id: str
    ) -> str:
        digest = sha256(task_id.encode("utf-8")).hexdigest()
        return _path_join(
            self._writes_shard_dir(thread_id, checkpoint_ns, checkpoint_id),
            f"w_{digest}{_CHECKPOINT_ARTIFACT_EXT}",
        )

    def _load_writes_map(
        self, thread_id: str, checkpoint_ns: str, checkpoint_id: str
    ) -> dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]]:
        merged: dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]] = {}
        legacy = self._writes_legacy_path(thread_id, checkpoint_ns, checkpoint_id)
        if self.fs.exists(legacy):
            with self.fs.open(legacy, "rb") as f:
                try:
                    merged.update(_unpack_writes_map_file(f.read()))
                except (ValueError, struct.error, OSError):
                    pass
        shard_dir = self._writes_shard_dir(thread_id, checkpoint_ns, checkpoint_id)
        if self.fs.exists(shard_dir):
            try:
                entries = self.fs.ls(shard_dir, detail=False)
            except OSError:
                entries = []
            for entry in entries:
                name = entry.rsplit("/", 1)[-1]
                if not name.endswith(_CHECKPOINT_ARTIFACT_EXT) or not name.startswith("w_"):
                    continue
                with self.fs.open(entry, "rb") as f:
                    try:
                        shard = _unpack_writes_map_file(f.read())
                    except (ValueError, struct.error, OSError):
                        continue
                merged.update(shard)
        return merged

    def _load_task_writes_shard(
        self, shard_path: str
    ) -> dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]]:
        if not self.fs.exists(shard_path):
            return {}
        with self.fs.open(shard_path, "rb") as f:
            try:
                return _unpack_writes_map_file(f.read())
            except (ValueError, struct.error, OSError):
                return {}

    def _save_task_writes_shard(
        self,
        shard_path: str,
        data: dict[tuple[str, int], tuple[str, str, tuple[str, bytes], str]],
    ) -> None:
        with self.fs.open(shard_path, "wb") as f:
            f.write(_pack_writes_map_file(data))

    def _load_blobs(
        self, thread_id: str, checkpoint_ns: str, versions: ChannelVersions
    ) -> dict[str, Any]:
        channel_values: dict[str, Any] = {}
        for k, v in versions.items():
            bp = self._blob_path(thread_id, checkpoint_ns, k, v)
            if not self.fs.exists(bp):
                continue
            with self.fs.open(bp, "rb") as f:
                try:
                    typed = _unpack_blob_file(f.read())
                except (ValueError, struct.error, OSError):
                    continue
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
        metadata: CheckpointMetadata | None = None,
    ) -> CheckpointTuple:
        checkpoint_bytes, metadata_bytes, parent_checkpoint_id = saved
        writes_map = self._load_writes_map(thread_id, checkpoint_ns, checkpoint_id)
        writes = writes_map.values()
        checkpoint_: Checkpoint = self.serde.loads_typed(checkpoint_bytes)
        metadata_ = metadata if metadata is not None else self.serde.loads_typed(metadata_bytes)
        return CheckpointTuple(
            config=response_config,
            checkpoint={
                **checkpoint_,
                "channel_values": self._load_blobs(
                    thread_id, checkpoint_ns, checkpoint_["channel_versions"]
                ),
            },
            metadata=metadata_,
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
                saved = _unpack_cpt_file(f.read())
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
            if name.endswith(_CHECKPOINT_ARTIFACT_EXT):
                ids.append(name[: -len(_CHECKPOINT_ARTIFACT_EXT)])
        if not ids:
            return None
        checkpoint_id = max(ids)
        path = self._cpt_path(thread_id, checkpoint_ns, checkpoint_id)
        with self.fs.open(path, "rb") as f:
            saved = _unpack_cpt_file(f.read())
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
        checkpoints_base = _path_join(self.root, _CHECKPOINT_TREE_DIR)
        if config:
            thread_ids = [config["configurable"]["thread_id"]]
        else:
            if not self.fs.exists(checkpoints_base):
                return
            thread_ids = []
            for e in self.fs.ls(checkpoints_base, detail=False):
                seg = e.rstrip("/").rsplit("/", 1)[-1]
                thread_ids.append(_decoder_segment(seg))

        config_checkpoint_ns = config["configurable"].get("checkpoint_ns") if config else None
        config_checkpoint_id = get_checkpoint_id(config) if config else None

        for thread_id in thread_ids:
            troot = _path_join(checkpoints_base, _encoder_segment(thread_id), "ns")
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
                    if not fname.endswith(_CHECKPOINT_ARTIFACT_EXT):
                        continue
                    cid = fname[: -len(_CHECKPOINT_ARTIFACT_EXT)]
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
                        saved = _unpack_cpt_file(f.read())
                    parsed_metadata: CheckpointMetadata | None = None
                    if filter:
                        parsed_metadata = self.serde.loads_typed(saved[1])
                        if not all(
                            query_value == parsed_metadata.get(query_key)
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
                        metadata=parsed_metadata,
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
                f.write(_pack_blob_file(blob))
        saved = (
            self.serde.dumps_typed(c),
            self.serde.dumps_typed(get_checkpoint_metadata(config, metadata)),
            config["configurable"].get("checkpoint_id"),
        )
        cpp = self._cpt_path(thread_id, checkpoint_ns, checkpoint["id"])
        with self.fs.open(cpp, "wb") as f:
            f.write(_pack_cpt_file(saved))
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
        shard_path = self._task_writes_shard_path(thread_id, checkpoint_ns, checkpoint_id, task_id)
        with _locked_task_writes_shard(shard_path):
            writes_map = self._load_task_writes_shard(shard_path)
            for idx, (c, v) in enumerate(writes):
                inner_key = (task_id, WRITES_IDX_MAP.get(c, idx))
                if inner_key[1] >= 0 and inner_key in writes_map:
                    continue
                writes_map[inner_key] = (
                    task_id,
                    c,
                    self.serde.dumps_typed(v),
                    task_path,
                )
            self._save_task_writes_shard(shard_path, writes_map)

    def delete_thread(self, thread_id: str) -> None:
        tdir = _path_join(self.root, _CHECKPOINT_TREE_DIR, _encoder_segment(thread_id))
        if self.fs.exists(tdir):
            self.fs.rm(tdir, recursive=True)

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        tuples = await asyncio.to_thread(
            list,
            self.list(config, filter=filter, before=before, limit=limit),
        )
        for item in tuples:
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)

    def get_next_version(
        self,
        current: str | int | float | None,
        channel: ChannelProtocol[Any, Any, Any] | None,
    ) -> str:
        if current is None:
            current_v = 0
        elif isinstance(current, (int, float)):
            current_v = int(current)
        else:
            current_v = int(str(current).split(".")[0])
        next_v = current_v + 1
        # Fixed-width decimal suffix (not ``random.random()`` + float format, which inserts an
        # extra dot and breaks zero-padding for some values).
        suffix = random.randrange(10**16)
        return f"{next_v:032}.{suffix:016d}"


def _normalize_checkpoint_base(checkpoint_base: str | None) -> str | None:
    if checkpoint_base is None:
        return None
    s = checkpoint_base.strip()
    return s or None


def _resolved_checkpoint_root(checkpoint_base: str | None) -> str:
    """Resolve saver root; null/empty ``checkpoint_base`` uses bare ``dr://``.

    With :class:`datarobot.fs.DataRobotFileSystem`, that resolves checkpoints under
    ``<catalog_id>/langgraph_checkpoints/checkpoints/`` (see
    :class:`DataRobotFileSystemCheckpointSaver`).
    """
    normalized = _normalize_checkpoint_base(checkpoint_base)
    if not normalized:
        return "dr://"
    return _normalize_dr_fs_root(normalized)


def default_langgraph_checkpointer(
    *, checkpoint_base: str | None = None
) -> DataRobotFileSystemCheckpointSaver:
    """Return a process-wide default saver using :class:`datarobot.fs.DataRobotFileSystem`.

    ``checkpoint_base`` is the optional ``dr://`` prefix for checkpoint files (typically passed
    from application configuration). If unset or empty, the saver root is ``dr://``, which the
    filesystem resolves to ``<catalog_id>/langgraph_checkpoints``; this module then writes under
    ``.../checkpoints/`` (see :class:`DataRobotFileSystemCheckpointSaver`). Best-effort removal
    of each distinct ``<effective_prefix>/checkpoints`` tree is registered with :mod:`atexit` (not
    the entire catalog prefix). For checkpoints that must survive process shutdown, construct
    and pass your own ``checkpointer=`` instead of relying on this default.
    """
    root = _resolved_checkpoint_root(checkpoint_base)

    if (existing := _default_process_checkpointers.get(root)) is not None:
        return existing
    from datarobot.fs import DataRobotFileSystem

    fs = DataRobotFileSystem()
    saver = DataRobotFileSystemCheckpointSaver(fs=fs, root=root)
    _default_process_checkpointers[saver.root] = saver
    _register_checkpoint_root_cleanup(fs, saver.root)
    return saver


def reset_default_langgraph_checkpointer_for_tests() -> None:
    """Clear process-wide defaults (for test isolation)."""
    _default_process_checkpointers.clear()
    _cleanup_registered_roots.clear()
    _task_writes_shard_locks.clear()
