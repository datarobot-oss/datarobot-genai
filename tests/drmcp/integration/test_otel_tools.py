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

"""Integration tests for DataRobot OTel MCP tools.

Real MCP protocol over stdio with stubbed REST (plan §7 tier 2): proves
registration, the generated JSON schema, and serialization. Truncation/dedup
correctness against realistic (oversized) traces is tier 1's job
(``tests/drmcp/unit/otel_tools/``) — this module's stub trace is deliberately
small, with just enough structure (one byte-identical duplicate group) to
exercise the dedup/drop-reporting path end to end over the real protocol.
"""

import json

import pytest
from mcp.types import TextContent

from datarobot_genai.drmcp.test_utils.mcp_utils_integration import integration_test_mcp_session
from datarobot_genai.drmcp.test_utils.mcp_utils_integration import (
    integration_test_server_params_with_env,
)
from datarobot_genai.drmcp.test_utils.stubs.otel_stubs import STUB_OTEL_EMPTY_ENTITY_ID
from datarobot_genai.drmcp.test_utils.stubs.otel_stubs import STUB_OTEL_ENTITY_ID
from datarobot_genai.drmcp.test_utils.stubs.otel_stubs import STUB_OTEL_ENTITY_TYPE
from datarobot_genai.drmcp.test_utils.stubs.otel_stubs import STUB_OTEL_MISSING_SPAN_ID
from datarobot_genai.drmcp.test_utils.stubs.otel_stubs import STUB_OTEL_SPAN_ID_ERROR
from datarobot_genai.drmcp.test_utils.stubs.otel_stubs import STUB_OTEL_TRACE_ID

_EXPECTED_TOOLS = frozenset(
    {
        "otel_traces_list",
        "otel_trace_get",
        "otel_span_payload_get",
        "otel_logs_list",
        "otel_metrics_catalog_list",
        "otel_metrics_values_get",
        "otel_entity_stats_get",
    }
)


def _otel_server_params():
    """Return server params with OTel tools enabled."""
    return integration_test_server_params_with_env({"ENABLE_OTEL_TOOLS": "true"})


def _parse_result(result: object) -> dict:
    assert not getattr(result, "isError", True)
    content = getattr(result, "content", [])
    assert len(content) > 0
    assert isinstance(content[0], TextContent)
    return json.loads(content[0].text)


def _schema_enum(schema: dict) -> list:
    """Read an ``enum`` list from a JSON-schema property, direct or ``anyOf``-wrapped."""
    if "enum" in schema:
        return schema["enum"]
    for variant in schema.get("anyOf", []):
        if "enum" in variant:
            return variant["enum"]
    return []


@pytest.mark.asyncio
class TestMCPOtelToolsRegistration:
    """Verify OTel tools are registered, with the generated JSON schema shape."""

    async def test_tools_registered(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            result = await session.list_tools()
            tool_names = {t.name for t in result.tools}
            missing = _EXPECTED_TOOLS - tool_names
            assert not missing, f"otel tools not registered: {missing}"

    async def test_every_otel_tool_has_display_metadata(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            result = await session.list_tools()
            otel_tools = {t.name: t for t in result.tools if t.name in _EXPECTED_TOOLS}
            assert set(otel_tools) == _EXPECTED_TOOLS
            for name, tool in otel_tools.items():
                assert tool.description, f"{name} missing a description"

    async def test_entity_type_schema_is_an_enum_of_all_eight_types(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            result = await session.list_tools()
            tool = next(t for t in result.tools if t.name == "otel_traces_list")
            entity_type_schema = tool.inputSchema["properties"]["entity_type"]
            values = _schema_enum(entity_type_schema)
            assert set(values) == {
                "deployment",
                "use_case",
                "experiment_container",
                "custom_application",
                "workload",
                "execution_environment",
                "custom_job",
                "artifact",
            }

    async def test_otel_trace_get_view_schema_is_summary_or_payloads(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            result = await session.list_tools()
            tool = next(t for t in result.tools if t.name == "otel_trace_get")
            view_schema = tool.inputSchema["properties"]["view"]
            assert set(_schema_enum(view_schema)) == {"summary", "payloads"}

    async def test_otel_logs_list_level_schema_has_all_six_levels(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            result = await session.list_tools()
            tool = next(t for t in result.tools if t.name == "otel_logs_list")
            level_schema = tool.inputSchema["properties"]["level"]
            assert set(_schema_enum(level_schema)) == {
                "debug",
                "info",
                "warn",
                "warning",
                "error",
                "critical",
            }

    async def test_otel_metrics_values_get_source_schema(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            result = await session.list_tools()
            tool = next(t for t in result.tools if t.name == "otel_metrics_values_get")
            source_schema = tool.inputSchema["properties"]["source"]
            assert set(_schema_enum(source_schema)) == {"configured", "autocollected"}


@pytest.mark.asyncio
class TestMCPOtelTracesListIntegration:
    """Integration tests for otel_traces_list."""

    async def test_otel_traces_list_returns_traces(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "otel_traces_list",
                    {"entity_type": STUB_OTEL_ENTITY_TYPE, "entity_id": STUB_OTEL_ENTITY_ID},
                )
            )
            assert "traces" in data
            assert data["count"] >= 1
            trace_ids = [t["trace_id"] for t in data["traces"]]
            assert STUB_OTEL_TRACE_ID in trace_ids

    async def test_otel_traces_list_invalid_entity_id(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            result = await session.call_tool(
                "otel_traces_list",
                {"entity_type": STUB_OTEL_ENTITY_TYPE, "entity_id": "not-a-hex-id"},
            )
            assert result.isError

    async def test_otel_traces_list_rejects_comma_joined_tools(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            result = await session.call_tool(
                "otel_traces_list",
                {
                    "entity_type": STUB_OTEL_ENTITY_TYPE,
                    "entity_id": STUB_OTEL_ENTITY_ID,
                    "tools": ["search,fetch"],
                },
            )
            assert result.isError


@pytest.mark.asyncio
class TestMCPOtelTraceGetIntegration:
    """Integration tests for otel_trace_get."""

    async def test_otel_trace_get_summary_view(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "otel_trace_get",
                    {
                        "entity_type": STUB_OTEL_ENTITY_TYPE,
                        "entity_id": STUB_OTEL_ENTITY_ID,
                        "trace_id": STUB_OTEL_TRACE_ID,
                    },
                )
            )
            assert data["trace_id"] == STUB_OTEL_TRACE_ID
            assert data["truncation"]["mode"] == "summary"
            assert data["truncation"]["payloads_omitted"] is True
            span_ids = [s["span_id"] for s in data["spans"]]
            assert STUB_OTEL_SPAN_ID_ERROR in span_ids
            error_span = next(s for s in data["spans"] if s["span_id"] == STUB_OTEL_SPAN_ID_ERROR)
            # prompt/completion + the duplicate group all count toward payload_chars
            # in the pre-dedup summary accounting.
            assert error_span["payload_chars"] > 0
            assert "completion" in error_span["payload_fields"]

    async def test_otel_trace_get_payloads_view_reports_duplicate_drop(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "otel_trace_get",
                    {
                        "entity_type": STUB_OTEL_ENTITY_TYPE,
                        "entity_id": STUB_OTEL_ENTITY_ID,
                        "trace_id": STUB_OTEL_TRACE_ID,
                        "view": "payloads",
                    },
                )
            )
            assert data["truncation"]["mode"] == "payloads"
            assert "spans_returned" in data["truncation"]
            assert "spans_dropped" in data["truncation"]
            error_span = next(s for s in data["spans"] if s["span_id"] == STUB_OTEL_SPAN_ID_ERROR)
            dropped = error_span["truncation"]["dropped_as_duplicate"]
            assert "gen_ai.task.output" in dropped
            assert "traceloop.entity.output" in dropped
            assert error_span["attributes"]["completion"]

    async def test_otel_trace_get_invalid_trace_id_length(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            result = await session.call_tool(
                "otel_trace_get",
                {
                    "entity_type": STUB_OTEL_ENTITY_TYPE,
                    "entity_id": STUB_OTEL_ENTITY_ID,
                    "trace_id": "too-short",
                },
            )
            assert result.isError


@pytest.mark.asyncio
class TestMCPOtelSpanPayloadGetIntegration:
    """Integration tests for otel_span_payload_get."""

    async def test_default_fields_reports_dropped_as_duplicate(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "otel_span_payload_get",
                    {
                        "entity_type": STUB_OTEL_ENTITY_TYPE,
                        "entity_id": STUB_OTEL_ENTITY_ID,
                        "trace_id": STUB_OTEL_TRACE_ID,
                        "span_id": STUB_OTEL_SPAN_ID_ERROR,
                    },
                )
            )
            assert data["span_id"] == STUB_OTEL_SPAN_ID_ERROR
            assert data["status_code"] == "ERROR"
            assert data["completion"]
            dropped = data["truncation"]["dropped_as_duplicate"]
            assert "gen_ai.task.output" in dropped
            assert "traceloop.entity.output" in dropped

    async def test_explicit_fields_bypasses_dedup(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "otel_span_payload_get",
                    {
                        "entity_type": STUB_OTEL_ENTITY_TYPE,
                        "entity_id": STUB_OTEL_ENTITY_ID,
                        "trace_id": STUB_OTEL_TRACE_ID,
                        "span_id": STUB_OTEL_SPAN_ID_ERROR,
                        "fields": ["gen_ai.task.output"],
                    },
                )
            )
            assert data["attributes"]["gen_ai.task.output"]
            assert not data["truncation"]["dropped_as_duplicate"]

    async def test_unknown_field_name_is_reported_not_found(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "otel_span_payload_get",
                    {
                        "entity_type": STUB_OTEL_ENTITY_TYPE,
                        "entity_id": STUB_OTEL_ENTITY_ID,
                        "trace_id": STUB_OTEL_TRACE_ID,
                        "span_id": STUB_OTEL_SPAN_ID_ERROR,
                        "fields": ["nonexistent.field"],
                    },
                )
            )
            assert data["truncation"]["fields_not_found"] == ["nonexistent.field"]

    async def test_span_not_found_is_a_tool_error(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            result = await session.call_tool(
                "otel_span_payload_get",
                {
                    "entity_type": STUB_OTEL_ENTITY_TYPE,
                    "entity_id": STUB_OTEL_ENTITY_ID,
                    "trace_id": STUB_OTEL_TRACE_ID,
                    "span_id": STUB_OTEL_MISSING_SPAN_ID,
                },
            )
            assert result.isError


@pytest.mark.asyncio
class TestMCPOtelLogsListIntegration:
    """Integration tests for otel_logs_list."""

    async def test_otel_logs_list_returns_logs(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "otel_logs_list",
                    {"entity_type": STUB_OTEL_ENTITY_TYPE, "entity_id": STUB_OTEL_ENTITY_ID},
                )
            )
            assert "logs" in data
            assert data["count"] >= 1
            assert data["logs"][0]["trace_id"] == STUB_OTEL_TRACE_ID

    async def test_otel_logs_list_never_carries_total_count(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "otel_logs_list",
                    {"entity_type": STUB_OTEL_ENTITY_TYPE, "entity_id": STUB_OTEL_ENTITY_ID},
                )
            )
            assert "total_count" not in data

    async def test_otel_logs_list_accepts_warning_and_critical(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            for level in ("warning", "critical"):
                result = await session.call_tool(
                    "otel_logs_list",
                    {
                        "entity_type": STUB_OTEL_ENTITY_TYPE,
                        "entity_id": STUB_OTEL_ENTITY_ID,
                        "level": level,
                    },
                )
                assert not result.isError, f"level={level!r} should be accepted"

    async def test_otel_logs_list_invalid_level(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            result = await session.call_tool(
                "otel_logs_list",
                {
                    "entity_type": STUB_OTEL_ENTITY_TYPE,
                    "entity_id": STUB_OTEL_ENTITY_ID,
                    "level": "verbose",
                },
            )
            assert result.isError


@pytest.mark.asyncio
class TestMCPOtelMetricsToolsIntegration:
    """Integration tests for otel_metrics_catalog_list and otel_metrics_values_get."""

    async def test_otel_metrics_catalog_list(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "otel_metrics_catalog_list",
                    {"entity_type": STUB_OTEL_ENTITY_TYPE, "entity_id": STUB_OTEL_ENTITY_ID},
                )
            )
            assert data["count"] >= 1
            names = [m["otel_name"] for m in data["metrics"]]
            assert "gen_ai.tokens.total" in names

    async def test_otel_metrics_values_get_autocollected_default(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "otel_metrics_values_get",
                    {"entity_type": STUB_OTEL_ENTITY_TYPE, "entity_id": STUB_OTEL_ENTITY_ID},
                )
            )
            assert "metrics" in data
            names = [m["otel_name"] for m in data["metrics"]]
            assert "cpu.usage" in names

    async def test_otel_metrics_values_get_configured(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "otel_metrics_values_get",
                    {
                        "entity_type": STUB_OTEL_ENTITY_TYPE,
                        "entity_id": STUB_OTEL_ENTITY_ID,
                        "source": "configured",
                    },
                )
            )
            assert "metric_aggregations" in data
            names = [m["otel_name"] for m in data["metric_aggregations"]]
            assert "custom.latency" in names


@pytest.mark.asyncio
class TestMCPOtelEntityStatsGetIntegration:
    """Integration tests for otel_entity_stats_get."""

    async def test_entity_with_data(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "otel_entity_stats_get",
                    {"entity_type": STUB_OTEL_ENTITY_TYPE, "entity_id": STUB_OTEL_ENTITY_ID},
                )
            )
            assert data["has_otel_data"] is True
            assert data["span_count"] == 15
            assert data["user_count"] == 2
            assert len(data["by_user"]) == 2

    async def test_entity_without_data(self) -> None:
        async with integration_test_mcp_session(server_params=_otel_server_params()) as session:
            data = _parse_result(
                await session.call_tool(
                    "otel_entity_stats_get",
                    {
                        "entity_type": STUB_OTEL_ENTITY_TYPE,
                        "entity_id": STUB_OTEL_EMPTY_ENTITY_ID,
                    },
                )
            )
            assert data["has_otel_data"] is False
            assert data["span_count"] == 0
            assert data["by_user"] == []
