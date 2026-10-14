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

"""Acceptance tests for OTel MCP tools (plan §7 tier 3).

Unlike the other tiers, this one measures tool-*description* quality: whether a
Sonnet-class model reaches for the right tool, and the right variant of that tool,
from a plain-English question -- without churning through the expensive option
first. The three cases below are the ones plan §7 names explicitly:

* "why did this trace fail?" -> otel_traces_list(status="error") then
  otel_trace_get, and specifically NOT otel_trace_get(view="payloads") on that
  first hop into the trace -- summary is enough to name the failing span; the
  much larger payloads view is not the tool this question calls for.
* "show me the LLM output on the failing span" -> otel_span_payload_get, not a
  second otel_trace_get -- when the trace and span are already known, drilling in
  directly is the right move, not re-fetching the whole trace again.
* "any errors in the logs for this deployment?" -> otel_logs_list(level="error").

These need an entity that is actually failing to be meaningful (an entity with no
OTel data, or no error traces, would fail these tests for a data reason, not a
tool-description reason). Rather than depend on a real one existing somewhere,
the ``otel_acceptance_entity`` fixture in conftest.py provisions its own: a Use
Case (``experiment_container``) into which it OTLP-exports a small failing
agentic run -- an ``agent.run`` whose ``llm.chat`` span errored with a 429 and
carries LLM output, plus an error-level log line correlated to that span --
waits for the REST API to read it back, and deletes it at session end (see
``tests/drmcp/helpers/otel_entity.py``). ``TEST_OTEL_ENTITY_ID`` opts back into
running against a real, externally-instrumented entity instead.
"""

import inspect
import os
from typing import Any

import pytest

from datarobot_genai.drmcp import ETETestExpectations
from datarobot_genai.drmcp import ToolBaseE2E
from datarobot_genai.drmcp import ToolCallTestExpectations
from datarobot_genai.drmcp import ete_test_mcp_session
from datarobot_genai.drmcp.test_utils.tool_base_ete import SHOULD_NOT_BE_EMPTY


@pytest.fixture(scope="session")
def expectations_for_why_did_this_trace_fail(
    otel_entity_type: str, otel_entity_id: str
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="otel_traces_list",
                parameters={
                    "entity_type": otel_entity_type,
                    "entity_id": otel_entity_id,
                    "status": "error",
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
            ToolCallTestExpectations(
                name="otel_trace_get",
                parameters={
                    "entity_type": otel_entity_type,
                    "entity_id": otel_entity_id,
                },
                result=SHOULD_NOT_BE_EMPTY,
                # The whole point of this case: otel_trace_get's summary view is
                # enough to name the failing span. An explicit view="payloads" on
                # this first hop means the description failed to steer the model
                # away from the much larger, budget-truncated view.
                forbidden_parameter_values={"view": "payloads"},
            ),
        ],
        llm_response_content_contains_expectations=[
            "error",
            "fail",
            "failed",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_show_llm_output_on_failing_span(
    otel_entity_type: str,
    otel_entity_id: str,
    otel_failing_trace_id: str,
    otel_failing_span_id: str,
) -> ETETestExpectations:
    return ETETestExpectations(
        # trace_id and span_id are already known, so otel_span_payload_get is the only
        # tool call this case expects -- but a legitimate paging continuation of that
        # same call (field_offset=... after a truncated field, which the tool's own
        # description invites) must stay allowed. What this case actually forbids, per
        # the plan's literal wording, is "a second otel_trace_get": re-fetching the
        # whole trace to get what the span_id already identifies is exactly the churn
        # this case exists to catch -- so that is asserted directly, rather than via a
        # blanket exact-tool-call-count that would also reject the legitimate
        # continuation call above.
        forbidden_tool_names=["otel_trace_get"],
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="otel_span_payload_get",
                parameters={
                    "entity_type": otel_entity_type,
                    "entity_id": otel_entity_id,
                    "trace_id": otel_failing_trace_id,
                    "span_id": otel_failing_span_id,
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "output",
            "completion",
            "response",
        ],
    )


@pytest.fixture(scope="session")
def expectations_for_logs_errors_for_entity(
    otel_entity_type: str, otel_entity_id: str
) -> ETETestExpectations:
    return ETETestExpectations(
        tool_calls_expected=[
            ToolCallTestExpectations(
                name="otel_logs_list",
                parameters={
                    "entity_type": otel_entity_type,
                    "entity_id": otel_entity_id,
                    "level": "error",
                },
                result=SHOULD_NOT_BE_EMPTY,
            ),
        ],
        llm_response_content_contains_expectations=[
            "error",
            "log",
        ],
    )


@pytest.mark.skipif(
    not os.getenv("ENABLE_OTEL_TOOLS"),
    reason="OTel tools are not enabled on the MCP server",
)
@pytest.mark.asyncio
class TestOtelToolsE2E(ToolBaseE2E):
    """End-to-end acceptance tests for OTel MCP tools."""

    async def test_why_did_this_trace_fail(
        self,
        llm_client: Any,
        otel_entity_type: str,
        otel_entity_id: str,
        expectations_for_why_did_this_trace_fail: ETETestExpectations,
    ) -> None:
        """LLM diagnoses a failing trace via otel_traces_list then otel_trace_get(summary)."""
        prompt = (
            f"Something went wrong recently with {otel_entity_type} '{otel_entity_id}'. "
            "Why did one of its OTel traces fail? Investigate and explain the root cause."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_why_did_this_trace_fail"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_why_did_this_trace_fail,
                llm_client,
                session,
                test_name,
            )

    async def test_show_llm_output_on_failing_span(
        self,
        llm_client: Any,
        otel_entity_type: str,
        otel_entity_id: str,
        otel_failing_trace_id: str,
        otel_failing_span_id: str,
        expectations_for_show_llm_output_on_failing_span: ETETestExpectations,
    ) -> None:
        """LLM fetches the LLM output on an already-identified failing span directly."""
        prompt = (
            f"In OTel trace '{otel_failing_trace_id}' for {otel_entity_type} "
            f"'{otel_entity_id}', span '{otel_failing_span_id}' is the one that errored. "
            "Show me the LLM output (the completion text) that span produced."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_show_llm_output_on_failing_span"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_show_llm_output_on_failing_span,
                llm_client,
                session,
                test_name,
            )

    async def test_logs_errors_for_entity(
        self,
        llm_client: Any,
        otel_entity_type: str,
        otel_entity_id: str,
        expectations_for_logs_errors_for_entity: ETETestExpectations,
    ) -> None:
        """LLM checks for error-level OTel logs via otel_logs_list(level="error")."""
        prompt = (
            f"Are there any errors in the OTel logs for {otel_entity_type} "
            f"'{otel_entity_id}'? If so, summarize what's failing."
        )
        async with ete_test_mcp_session() as session:
            frame = inspect.currentframe()
            test_name = frame.f_code.co_name if frame else "test_logs_errors_for_entity"
            await self._run_test_with_expectations(
                prompt,
                expectations_for_logs_errors_for_entity,
                llm_client,
                session,
                test_name,
            )
