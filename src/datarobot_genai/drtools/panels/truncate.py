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

"""Structure-preserving truncation for LLM-facing panel payloads (from wren-mcp)."""

from __future__ import annotations

from typing import Any


def truncate_for_llm(
    data: Any,
    *,
    max_string_length: int = 200,
    max_array_items: int = 5,
    max_object_keys: int = 30,
    max_depth: int = 6,
    _current_depth: int = 0,
) -> Any:
    """Truncate nested data for LLM consumption while preserving structure.

    Arrays keep the first ``max_array_items`` entries (plus a remaining-count
    marker), strings are capped at ``max_string_length`` chars, objects keep the
    first ``max_object_keys`` keys, and nesting beyond ``max_depth`` collapses to
    a one-line summary. Primitives pass through unchanged.
    """
    if _current_depth >= max_depth:
        return _summarize_truncated(data)

    if isinstance(data, str):
        if len(data) > max_string_length:
            return data[:max_string_length] + f"... ({len(data) - max_string_length} more chars)"
        return data

    recurse_kwargs = {
        "max_string_length": max_string_length,
        "max_array_items": max_array_items,
        "max_object_keys": max_object_keys,
        "max_depth": max_depth,
        "_current_depth": _current_depth + 1,
    }

    if isinstance(data, list):
        truncated_list: list[Any] = [
            truncate_for_llm(item, **recurse_kwargs) for item in data[:max_array_items]
        ]
        if len(data) > max_array_items:
            truncated_list.append(f"... ({len(data) - max_array_items} more items)")
        return truncated_list

    if isinstance(data, dict):
        keys = list(data.keys())
        truncated_dict: dict[str, Any] = {
            key: truncate_for_llm(data[key], **recurse_kwargs) for key in keys[:max_object_keys]
        }
        if len(keys) > max_object_keys:
            truncated_dict["..."] = f"({len(keys) - max_object_keys} more keys)"
        return truncated_dict

    return data


def _summarize_truncated(data: Any) -> str | Any:
    """Create a brief summary for data truncated due to the depth limit."""
    if isinstance(data, dict):
        return f"{{...}} ({len(data)} keys)"
    if isinstance(data, list):
        return f"[...] ({len(data)} items)"
    if isinstance(data, str):
        if len(data) > 50:
            return f'"{data[:50]}..." ({len(data)} chars)'
        return data
    return data


def truncate_source_code(src: str, max_lines: int = 50) -> str:
    """Truncate source code to its first ``max_lines`` lines."""
    lines = src.splitlines()
    if len(lines) <= max_lines:
        return src
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
