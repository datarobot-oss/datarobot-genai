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

"""Shared query parsing and filtering helpers for paginated REST list routes."""

from typing import Any

from starlette.requests import Request


def parse_pagination(request: Request, *, default_limit: int = 100) -> tuple[int, int]:
    """Read ``limit`` and ``offset`` from the query string.

    Non-integer or negative values fall back to the defaults, so a malformed query never
    500s the route.
    """

    def _non_negative_int(name: str, default: int) -> int:
        raw = request.query_params.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return default
        return value if value >= 0 else default

    return _non_negative_int("limit", default_limit), _non_negative_int("offset", 0)


def parse_list_filters(
    request: Request,
) -> tuple[str | None, list[str] | None, list[str] | None]:
    """Read optional ``name``, ``provider`` and ``category`` filters from the query string.

    ``name`` is a single exact match. ``provider`` and ``category`` are **multi-valued**,
    mirroring the multi-select checkboxes in the gallery filter panel. Both forms are
    accepted and combined:
      - comma-separated: ``?category=dr_connectors,dr_web_search``
      - repeated params: ``?category=dr_connectors&category=dr_web_search``
    Blank tokens are dropped; a dimension with no non-blank values is treated as absent.
    """

    def _single(key: str) -> str | None:
        raw = request.query_params.get(key)
        if raw is None:
            return None
        raw = raw.strip()
        return raw or None

    def _multi(key: str) -> list[str] | None:
        values = [
            token.strip()
            for raw in request.query_params.getlist(key)
            for token in raw.split(",")
            if token.strip()
        ]
        return values or None

    return _single("name"), _multi("provider"), _multi("category")


def apply_list_filters(
    items: list[dict[str, Any]],
    name: str | None,
    providers: list[str] | None,
    categories: list[str] | None,
) -> list[dict[str, Any]]:
    """Filter list *items* by ``name``, ``provider`` and/or ``category``.

    ``provider`` and ``category`` are match-any within the dimension; dimensions combine
    with **AND**. Unrecognised filter values simply match nothing.
    """
    if name is not None:
        items = [item for item in items if item.get("name") == name]
    if providers is not None:
        wanted = set(providers)
        items = [item for item in items if item.get("provider") in wanted]
    if categories is not None:
        wanted = set(categories)
        items = [
            item
            for item in items
            if wanted.intersection(category["name"] for category in item.get("categories") or ())
        ]
    return items


def enum_items(mapping: dict[Any, str]) -> list[dict[str, str]]:
    """Serialise an ordered ``value -> label`` map into ``[{"value", "label"}]`` items."""
    return [{"value": str(value), "label": label} for value, label in mapping.items()]
