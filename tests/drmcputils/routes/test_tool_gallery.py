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

"""FastMCP-free tests for the shared tool gallery route helpers."""

import importlib
import importlib.util
import pkgutil

from datarobot_genai.drmcputils.tool_gallery import DRTOOLS_PRIVATE_METADATA_KEYS
from datarobot_genai.drmcputils.tool_gallery import is_hosted
from datarobot_genai.drmcputils.tool_gallery import merge_tool_info
from datarobot_genai.drtools.core import get_registered_tools


def _all_live_drtools_metadata() -> list[tuple[str, dict]]:
    """Import every drtools tool module and return (tool_name, metadata) pairs."""

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
    return [((md.get("name") or fn.__name__), md) for fn, md in get_registered_tools()]


class _FakeTool:
    def __init__(self, name: str, tags: set[str] | None = None, meta: dict | None = None) -> None:
        self.name = name
        self.tags = tags or set()
        self.meta = meta


class TestIsHosted:
    def test_user_tool_deployment_is_hosted(self) -> None:
        assert is_hosted(_FakeTool("t", meta={"tool_category": "USER_TOOL_DEPLOYMENT"}))

    def test_built_in_tool_is_not_hosted(self) -> None:
        assert not is_hosted(_FakeTool("t", meta={"tool_category": "BUILT_IN_TOOL"}))

    def test_no_meta_is_not_hosted(self) -> None:
        assert not is_hosted(_FakeTool("t", meta=None))

    def test_empty_meta_is_not_hosted(self) -> None:
        assert not is_hosted(_FakeTool("t", meta={}))


class TestMergeToolInfo:
    def test_merges_ui_metadata_and_derived_categories(self) -> None:
        tool = _FakeTool("jira_search_issues", tags={"jira", "search"})
        ui = {
            "jira_search_issues": {
                "display_name": "Jira — Search Issues",
                "description_ui": "Find Jira issues matching a JQL query.",
                "auth_provider": "jira",
            }
        }
        merged = merge_tool_info(tool, ui)
        assert merged["name"] == "jira_search_issues"
        assert merged["display_name"] == "Jira — Search Issues"
        assert merged["description_ui"] == "Find Jira issues matching a JQL query."
        assert merged["auth_provider"] == "jira"
        assert merged["tags"] == ["jira", "search"]
        assert merged["categories"] == ["dr_connector_jira", "dr_connectors"]
        assert merged["hosted"] is False

    def test_tool_without_ui_metadata_gets_none_fields(self) -> None:
        merged = merge_tool_info(_FakeTool("jira_search_issues"), {})
        assert merged["display_name"] is None
        assert merged["description_ui"] is None
        assert merged["auth_provider"] is None
        # Categories still derived from the static taxonomy.
        assert merged["categories"] == ["dr_connector_jira", "dr_connectors"]

    def test_hosted_tool_has_no_categories(self) -> None:
        tool = _FakeTool("user_xyz", meta={"tool_category": "USER_TOOL_DEPLOYMENT"})
        merged = merge_tool_info(tool, {})
        assert merged["hosted"] is True
        assert merged["categories"] == []


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

        # Ensure modules are imported.
        _all_live_drtools_metadata()
        missing = []
        for fn, md in get_registered_tools():
            mod = fn.__module__
            pkg = mod.split(".")[3] if mod.startswith("datarobot_genai.drtools.") else ""
            if pkg in needs_auth and not md.get("auth_provider"):
                missing.append(md.get("name") or fn.__name__)
        assert not missing, f"connector/web tools missing auth_provider: {sorted(missing)}"


class TestUiMetadataNeverLeaksToMcpClient:
    """The UI fields must be in the strip set so agents/LLMs never see them."""

    def test_ui_keys_are_private(self) -> None:
        for key in ("display_name", "description_ui", "auth_provider"):
            assert key in DRTOOLS_PRIVATE_METADATA_KEYS, (
                f"{key!r} must be stripped before FastMCP registration"
            )
