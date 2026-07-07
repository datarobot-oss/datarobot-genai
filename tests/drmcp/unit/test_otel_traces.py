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

from datarobot_genai.drmcp.test_utils import otel_traces


class TestDeploymentIdFromUrl:
    def test_direct_access_url(self) -> None:
        url = "https://staging.datarobot.com/api/v2/deployments/6a39fdf6e42e193c806d7587/directAccess/mcp"
        assert otel_traces.deployment_id_from_url(url) == "6a39fdf6e42e193c806d7587"

    def test_local_url(self) -> None:
        assert otel_traces.deployment_id_from_url("http://localhost:8652/mcp/") is None

    def test_none(self) -> None:
        assert otel_traces.deployment_id_from_url(None) is None


class TestFormatTracesTable:
    def test_empty(self) -> None:
        assert "no traces" in otel_traces.format_traces_table([])

    def test_table_contains_tools_and_duration(self) -> None:
        traces = [
            {
                "traceId": "cb7937ddeed453d236c70aa1f4e19f33",
                "spansCount": 4,
                "errorSpansCount": 4,
                "duration": 90909.0,  # ms
                "rootSpanName": "mcp.request.tools/call",
                "timestamp": 1783442017547.0,  # epoch ms
                "tools": [{"name": "filter_panel", "callCount": 3}],
            },
            {
                "traceId": "88d7f331281bd65ebcfdc50dbc808402",
                "spansCount": 1,
                "errorSpansCount": 0,
                "duration": 12.0,
                "rootSpanName": "mcp.request.initialize",
                "timestamp": 1783442017180.0,
                "tools": [],
            },
        ]
        table = otel_traces.format_traces_table(traces)
        assert "mcp.request.tools/call" in table
        assert "90.9s" in table
        assert "filter_panel×3" in table
        assert "12ms" in table
        # header present
        assert "ROOT SPAN" in table.splitlines()[0]


class TestFormatTraceTree:
    def test_tree_nesting_and_status(self) -> None:
        trace = {
            "traceId": "cb7937ddeed453d236c70aa1f4e19f33",
            "spans": [
                {
                    "spanId": "root1",
                    "parentSpanId": None,
                    "name": "mcp.request.tools/call",
                    "statusCode": "Error",
                    "statusMessage": "ToolError: sandbox timed out",
                    "duration": 90_909_926_146.0,  # ns
                    "startTime": 1.0,
                    "attributes": {"mcp.method.name": "tools/call"},
                },
                {
                    "spanId": "child1",
                    "parentSpanId": "root1",
                    "name": "execute_tool",
                    "statusCode": "Ok",
                    "duration": 1_500_000.0,  # ns -> 1.5ms -> "2ms"
                    "startTime": 2.0,
                    "attributes": {"mcp.tool.name": "filter_panel"},
                },
            ],
        }
        tree = otel_traces.format_trace_tree(trace)
        lines = tree.splitlines()
        assert lines[0].startswith("trace cb7937")
        assert "✗ mcp.request.tools/call" in tree
        assert "[90.9s]" in tree
        assert "ToolError: sandbox timed out" in tree
        # child is indented deeper than its parent and carries the tool name
        child_line = next(line for line in lines if "execute_tool" in line)
        parent_line = next(line for line in lines if "mcp.request.tools/call" in line)
        assert len(child_line) - len(child_line.lstrip()) > len(parent_line) - len(
            parent_line.lstrip()
        )
        assert "tool=filter_panel" in child_line

    def test_orphan_parent_treated_as_root(self) -> None:
        trace = {
            "traceId": "t",
            "spans": [
                {
                    "spanId": "s1",
                    "parentSpanId": "external-client-span",
                    "name": "mcp.request.tools/call",
                    "statusCode": "Ok",
                    "duration": 1e9,
                    "startTime": 1.0,
                }
            ],
        }
        tree = otel_traces.format_trace_tree(trace)
        assert "✓ mcp.request.tools/call" in tree

    def test_empty(self) -> None:
        assert "no spans" in otel_traces.format_trace_tree({"spans": []})
