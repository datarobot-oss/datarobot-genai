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

"""Unit tests for drmcpbase.routes.helpers."""

from typing import Any

from starlette.requests import Request

from datarobot_genai.drmcpbase.routes.helpers import apply_list_filters
from datarobot_genai.drmcpbase.routes.helpers import parse_list_filters
from datarobot_genai.drmcpbase.routes.helpers import parse_pagination


def _request(query: str = "") -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/static/tools/",
        "query_string": query.encode(),
        "headers": [],
    }
    return Request(scope)


class TestParsePagination:
    def test_defaults(self) -> None:
        limit, offset = parse_pagination(_request())
        assert limit == 100
        assert offset == 0

    def test_malformed_falls_back(self) -> None:
        limit, offset = parse_pagination(_request("limit=abc&offset=xyz"))
        assert limit == 100
        assert offset == 0

    def test_negative_falls_back(self) -> None:
        limit, offset = parse_pagination(_request("limit=-1&offset=-5"))
        assert limit == 100
        assert offset == 0


class TestParseGalleryFilters:
    def test_comma_and_repeat_combine(self) -> None:
        name, providers, categories = parse_list_filters(
            _request("category=dr_a,dr_b&category=dr_c&provider=datarobot,third_party")
        )
        assert name is None
        assert providers == ["datarobot", "third_party"]
        assert categories == ["dr_a", "dr_b", "dr_c"]

    def test_blank_dimensions_are_absent(self) -> None:
        name, providers, categories = parse_list_filters(_request("name=&provider=&category="))
        assert name is None
        assert providers is None
        assert categories is None


class TestApplyGalleryFilters:
    def test_category_match_any_within_dimension(self) -> None:
        items: list[dict[str, Any]] = [
            {
                "name": "a",
                "provider": "datarobot",
                "categories": [{"name": "dr_connectors", "label": "Connectors", "kind": "leaf"}],
            },
            {"name": "b", "provider": "third_party", "categories": []},
        ]
        filtered = apply_list_filters(items, None, None, ["dr_connectors"])
        assert [item["name"] for item in filtered] == ["a"]
