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

"""Guards that keep the live drtools registry aligned with drmcputils taxonomy/UI metadata.

These tests import every drtools tool module, so they belong in the drmcp job
(which installs the full drtools dependency set) rather than drmcputils.
"""

import importlib
import importlib.util
import pkgutil

from datarobot_genai.drmcputils.categories import LEAF_CATEGORY_TOOLS
from datarobot_genai.drtools.core import get_registered_tools


def _load_all_drtools_modules() -> None:
    """Import every drtools tool module so @tool_metadata decorators populate the registry."""

    def _walk(package_name: str) -> None:
        spec = importlib.util.find_spec(package_name)
        if spec is None or not spec.submodule_search_locations:
            return
        for info in pkgutil.iter_modules(spec.submodule_search_locations, package_name + "."):
            last = info.name.rsplit(".", 1)[-1]
            if last.startswith("_"):
                continue
            if package_name == "datarobot_genai.drtools" and last == "core":
                continue
            if info.ispkg:
                _walk(info.name)
            else:
                importlib.import_module(info.name)

    _walk("datarobot_genai.drtools")


def _all_live_drtools_tool_names() -> set[str]:
    _load_all_drtools_modules()
    return {(metadata.get("name") or func.__name__) for func, metadata in get_registered_tools()}


def _all_live_drtools_metadata() -> list[tuple[str, dict]]:
    _load_all_drtools_modules()
    return [((md.get("name") or fn.__name__), md) for fn, md in get_registered_tools()]


class TestTaxonomyCompleteness:
    """Guard against drift between live @tool_metadata tools and the taxonomy."""

    def test_every_live_tool_belongs_to_a_leaf_category(self) -> None:
        live = _all_live_drtools_tool_names()
        categorized: set[str] = set().union(*LEAF_CATEGORY_TOOLS.values())
        uncategorized = live - categorized
        assert not uncategorized, (
            "These live @tool_metadata tools are missing from LEAF_CATEGORY_TOOLS "
            f"and will be dropped by category filters: {sorted(uncategorized)}"
        )

    def test_no_stale_tool_names_in_taxonomy(self) -> None:
        live = _all_live_drtools_tool_names()
        categorized: set[str] = set().union(*LEAF_CATEGORY_TOOLS.values())
        stale = categorized - live
        assert not stale, (
            "These tool names appear in LEAF_CATEGORY_TOOLS but no longer have a "
            f"live @tool_metadata decorator: {sorted(stale)}"
        )


class TestDrtoolsUiMetadataCompleteness:
    """Every live drtools tool must carry the UI metadata the gallery surfaces."""

    def test_every_tool_has_display_name_and_description_ui(self) -> None:
        missing_display = []
        missing_desc = []
        for name, md in _all_live_drtools_metadata():
            if not md.get("display_name"):
                missing_display.append(name)
            if not md.get("description_ui"):
                missing_desc.append(name)
        assert not missing_display, f"tools missing display_name: {sorted(missing_display)}"
        assert not missing_desc, f"tools missing description_ui: {sorted(missing_desc)}"

    def test_connector_and_web_tools_have_auth_provider(self) -> None:
        needs_auth = {"confluence", "jira", "gdrive", "microsoft_graph", "perplexity", "tavily"}

        _load_all_drtools_modules()
        missing = []
        for fn, md in get_registered_tools():
            mod = fn.__module__
            pkg = mod.split(".")[3] if mod.startswith("datarobot_genai.drtools.") else ""
            if pkg in needs_auth and not md.get("auth_provider"):
                missing.append(md.get("name") or fn.__name__)
        assert not missing, f"connector/web tools missing auth_provider: {sorted(missing)}"
