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

from typing import Any

# Max page / batch size for predictive tools
DR_PREDICTIVE_API_PAGINATION_MAX = 100


def _clamp_limit(limit: int) -> tuple[int, str | None]:
    """Clamp page size to [1, DR_PREDICTIVE_API_PAGINATION_MAX] and an optional user-facing note."""
    if limit < 1:
        return (
            DR_PREDICTIVE_API_PAGINATION_MAX,
            f"""Limit must be at least 1. The maximum limit of """
            f"""{DR_PREDICTIVE_API_PAGINATION_MAX} was applied.""",
        )
    if limit > DR_PREDICTIVE_API_PAGINATION_MAX:
        return (
            DR_PREDICTIVE_API_PAGINATION_MAX,
            f"""Limit cannot exceed {DR_PREDICTIVE_API_PAGINATION_MAX}. """
            f"""The maximum limit of {DR_PREDICTIVE_API_PAGINATION_MAX} was applied.""",
        )
    return (limit, None)


def _merge_pagination_metadata(
    final_results: dict[str, Any],
    api_response: dict[str, Any] | list,
    message: str | None = None,
    *,
    offset: int | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Add offset/limit echo and DataRobot list pagination (next, previous, total) when present."""
    if offset is not None:
        final_results["offset"] = offset
    if limit is not None:
        final_results["limit"] = limit
    if message is not None:
        final_results["note"] = message
    if isinstance(api_response, dict):
        for key in ("next", "previous"):
            if key in api_response and api_response[key] is not None:
                final_results[key] = api_response[key]
        for total_key in ("total_count", "total"):
            if total_key in api_response and api_response[total_key] is not None:
                final_results["total_count"] = api_response[total_key]
                break
    return final_results
