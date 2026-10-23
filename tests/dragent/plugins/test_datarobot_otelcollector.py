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

"""Tests for the datarobot_otelcollector NAT telemetry exporter."""

from __future__ import annotations

import json
import uuid
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from ag_ui.core import TextMessageContentEvent
from ag_ui.core import TextMessageEndEvent
from ag_ui.core import TextMessageStartEvent
from nat.builder.context import IntermediateStep
from nat.builder.context import IntermediateStepPayload
from nat.builder.context import IntermediateStepType
from nat.builder.context import InvocationNode
from nat.data_models.intermediate_step import StreamEventData
from nat.data_models.span import Span
from nat.plugins.opentelemetry import OTLPSpanAdapterExporter
from pydantic import BaseModel

from datarobot_genai.core.telemetry.datarobot_otel import resolve_datarobot_headers_from_env
from datarobot_genai.dragent.frontends.response import DRAgentEventResponse
from datarobot_genai.dragent.plugins.datarobot_otelcollector import (
    DataRobotOtelCollectorTelemetryExporter,
)
from datarobot_genai.dragent.plugins.datarobot_otelcollector import DataRobotOTLPSpanAdapterExporter
from datarobot_genai.dragent.plugins.datarobot_otelcollector import DropValueObjAttributesProcessor
from datarobot_genai.dragent.plugins.datarobot_otelcollector import (
    datarobot_otelcollector_telemetry_exporter,
)

ENDPOINT = "https://staging.datarobot.com/otel/v1/traces"
ENTITY_ID = "deployment-abc123"
API_KEY = "test-key-do-not-use"


class _Payload(BaseModel):
    query: str


_ENV_VARS = (
    "DATAROBOT_API_TOKEN",
    "MLOPS_DEPLOYMENT_ID",
    "OTEL_ENTITY_ID",
    "MLOPS_RUNTIME_PARAM_OTEL_ENTITY_ID",
    "DATAROBOT_ENDPOINT",
    "DATAROBOT_PUBLIC_API_ENDPOINT",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_HEADERS",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Strip all DataRobot/MLOps env vars so default_factory paths see a
    deterministic empty environment. Individual tests opt-in to specific
    values with monkeypatch.setenv afterwards.
    """
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _make_config(**overrides) -> DataRobotOtelCollectorTelemetryExporter:
    base = {
        "endpoint": ENDPOINT,
        "project": "test-agent",
    }
    base.update(overrides)
    return DataRobotOtelCollectorTelemetryExporter(**base)


def _set_datarobot_env(clean_env, *, api_key=API_KEY, entity_id=ENTITY_ID):
    clean_env.setenv("DATAROBOT_API_TOKEN", api_key)
    clean_env.setenv("MLOPS_DEPLOYMENT_ID", entity_id.removeprefix("deployment-"))


class TestConfigValidation:
    def test_explicit_values_valid(self, clean_env):
        cfg = _make_config()
        assert cfg.endpoint == ENDPOINT
        assert cfg.extra_headers == {}


class TestEnvDerivation:
    def test_endpoint_derived_from_env(self, clean_env):
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == "https://example.test/otel/v1/traces"

    def test_public_endpoint_takes_priority_over_endpoint(self, clean_env):
        # resolve_otel_traces_endpoint_from_env puts DATAROBOT_PUBLIC_API_ENDPOINT
        # ahead of DATAROBOT_ENDPOINT — same precedence here.
        clean_env.setenv("DATAROBOT_PUBLIC_API_ENDPOINT", "https://public.test/api/v2")
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://internal.test/api/v2")
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == "https://public.test/otel/v1/traces"

    def test_endpoint_strips_api_path_segment(self, clean_env):
        # DATAROBOT_ENDPOINT typically includes the /api/v2 suffix; the OTel
        # collector lives at /otel/v1/traces off the host, not under /api.
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == "https://example.test/otel/v1/traces"

    def test_endpoint_empty_when_env_unset(self, clean_env):
        # No silent fallback to app.datarobot.com — empty is the explicit
        # "not configured" signal so an unconfigured env never targets
        # the public DR endpoint by accident.
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == ""

    def test_explicit_otlp_base_url_appends_traces_path(self, clean_env):
        # OTEL_EXPORTER_OTLP_ENDPOINT is the standard OTel base URL; we append
        # /v1/traces rather than returning it verbatim.
        clean_env.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.test:4318")
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == "https://collector.test:4318/v1/traces"

    def test_explicit_otlp_base_url_strips_trailing_slash(self, clean_env):
        clean_env.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.test:4318/")
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == "https://collector.test:4318/v1/traces"

    def test_explicit_otlp_endpoint_wins_over_datarobot_endpoint(self, clean_env):
        # The standard OTel override takes precedence over the DR-derived
        # endpoint so an operator can point spans at any collector.
        clean_env.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.test")
        clean_env.setenv("DATAROBOT_ENDPOINT", "https://example.test/api/v2")
        cfg = DataRobotOtelCollectorTelemetryExporter(project="test-agent")
        assert cfg.endpoint == "https://collector.test/v1/traces"


class TestHeaderResolvers:
    def test_headers_none_when_datarobot_env_incomplete(self, clean_env):
        assert resolve_datarobot_headers_from_env() is None

    def test_headers_none_when_api_key_only(self, clean_env):
        clean_env.setenv("DATAROBOT_API_TOKEN", API_KEY)
        assert resolve_datarobot_headers_from_env() is None

    def test_headers_none_when_entity_id_only(self, clean_env):
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        assert resolve_datarobot_headers_from_env() is None

    def test_headers_from_otel_entity_id_for_local_dev(self, clean_env):
        # For local dev, OTEL_ENTITY_ID is set via .env / Pulumi outputs and
        # loaded by _OtelSettings — mirrors config.py's otel_entity_id field.
        clean_env.setenv("DATAROBOT_API_TOKEN", API_KEY)
        clean_env.setenv("OTEL_ENTITY_ID", ENTITY_ID)
        headers = resolve_datarobot_headers_from_env()
        assert headers == {
            "X-DataRobot-Api-Key": API_KEY,
            "X-DataRobot-Entity-Id": ENTITY_ID,
        }

    def test_mlops_deployment_id_takes_priority_over_otel_entity_id(self, clean_env):
        clean_env.setenv("DATAROBOT_API_TOKEN", API_KEY)
        clean_env.setenv("MLOPS_DEPLOYMENT_ID", "abc123")
        clean_env.setenv("OTEL_ENTITY_ID", "deployment-other")
        headers = resolve_datarobot_headers_from_env()
        assert headers is not None
        assert headers["X-DataRobot-Entity-Id"] == "deployment-abc123"

    def test_headers_none_when_otel_entity_id_without_api_key(self, clean_env):
        clean_env.setenv("OTEL_ENTITY_ID", ENTITY_ID)
        assert resolve_datarobot_headers_from_env() is None


class TestExporterFactory:
    # The @register_telemetry_exporter decorator wraps the function in
    # asynccontextmanager, so each call below uses ``async with``.

    async def test_emits_both_datarobot_headers(self, clean_env):
        _set_datarobot_env(clean_env)
        cfg = _make_config()
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        mock_exporter.assert_called_once()
        kwargs = mock_exporter.call_args.kwargs
        assert kwargs["endpoint"] == ENDPOINT
        assert kwargs["headers"]["X-DataRobot-Api-Key"] == API_KEY
        assert kwargs["headers"]["X-DataRobot-Entity-Id"] == ENTITY_ID

    async def test_extra_headers_override_defaults(self, clean_env):
        _set_datarobot_env(clean_env)
        cfg = _make_config(
            extra_headers={
                "X-DataRobot-Entity-Id": "deployment-override",
                "X-Custom-Trace": "yes",
            },
        )
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        kwargs = mock_exporter.call_args.kwargs
        assert kwargs["headers"]["X-DataRobot-Entity-Id"] == "deployment-override"
        assert kwargs["headers"]["X-Custom-Trace"] == "yes"
        # API key still present because extra_headers didn't override it.
        assert kwargs["headers"]["X-DataRobot-Api-Key"] == API_KEY

    async def test_secret_value_extracted_for_header(self, clean_env):
        # The header dict should contain the resolved secret string, not a
        # SecretStr wrapper — OTLP exporter expects str values.
        _set_datarobot_env(clean_env)
        cfg = _make_config()
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        api_key_header = mock_exporter.call_args.kwargs["headers"]["X-DataRobot-Api-Key"]
        assert isinstance(api_key_header, str)
        assert api_key_header == API_KEY

    async def test_project_becomes_service_name(self, clean_env):
        # Resource attributes are independent of auth headers, but the exporter
        # factory still needs resolvable headers — supply DR env here.
        _set_datarobot_env(clean_env)
        # The inherited ``project`` field should populate the OTel
        # ``service.name`` resource attribute, matching NAT's built-in
        # otelcollector exporter.
        cfg = _make_config(project="my-agent")
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        attrs = mock_exporter.call_args.kwargs["resource_attributes"]
        assert attrs["service.name"] == "my-agent"
        assert attrs["telemetry.sdk.language"] == "python"
        assert attrs["telemetry.sdk.name"] == "opentelemetry"
        assert "telemetry.sdk.version" in attrs

    async def test_resource_attributes_override_defaults(self, clean_env):
        _set_datarobot_env(clean_env)
        # Explicit resource_attributes win over the defaults derived from
        # ``project`` / SDK metadata.
        cfg = _make_config(
            project="my-agent",
            resource_attributes={"service.name": "override-name", "env": "prod"},
        )
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        attrs = mock_exporter.call_args.kwargs["resource_attributes"]
        assert attrs["service.name"] == "override-name"
        assert attrs["env"] == "prod"
        # SDK defaults still merged in.
        assert attrs["telemetry.sdk.language"] == "python"


class TestOtlpHeadersEnvOverride:
    """When OTEL_EXPORTER_OTLP_HEADERS is set, the standard OTel header env var
    wins over the DataRobot auth headers derived from env.
    """

    async def test_headers_parsed_from_env(self, clean_env):
        clean_env.setenv(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "X-DataRobot-Api-Key=env-key,X-DataRobot-Entity-Id=deployment-env",
        )
        cfg = _make_config()
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        headers = mock_exporter.call_args.kwargs["headers"]
        # Env-provided headers replace the DR auth headers derived from env.
        assert headers["X-DataRobot-Api-Key"] == "env-key"
        assert headers["X-DataRobot-Entity-Id"] == "deployment-env"

    async def test_env_headers_take_precedence_over_datarobot_env(self, clean_env):
        # DR env still supplies API key + entity id, but the OTel header var
        # must override them entirely.
        _set_datarobot_env(clean_env)
        clean_env.setenv("OTEL_EXPORTER_OTLP_HEADERS", "X-DataRobot-Api-Key=env-key")
        cfg = _make_config()
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        headers = mock_exporter.call_args.kwargs["headers"]
        assert headers["X-DataRobot-Api-Key"] == "env-key"
        # The env-derived value did not leak through.
        assert headers["X-DataRobot-Api-Key"] != API_KEY

    async def test_extra_headers_still_merged_over_env_headers(self, clean_env):
        # extra_headers always win on collision, even over env-supplied headers.
        clean_env.setenv("OTEL_EXPORTER_OTLP_HEADERS", "X-DataRobot-Api-Key=env-key")
        cfg = _make_config(extra_headers={"X-DataRobot-Api-Key": "extra-key", "X-Custom": "v"})
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        headers = mock_exporter.call_args.kwargs["headers"]
        assert headers["X-DataRobot-Api-Key"] == "extra-key"
        assert headers["X-Custom"] == "v"

    async def test_header_value_with_equals_preserved(self, clean_env):
        # A header value can legitimately contain '=' (e.g. base64 padding or
        # a token with '='). Splitting on the first '=' only must keep the
        # full value intact rather than truncating at the first '='.
        clean_env.setenv(
            "OTEL_EXPORTER_OTLP_HEADERS",
            "Authorization=Basic dXNlcjpwYXNz==,X-Token=a=b=c",
        )
        cfg = _make_config()
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        headers = mock_exporter.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Basic dXNlcjpwYXNz=="
        assert headers["X-Token"] == "a=b=c"

    async def test_datarobot_env_headers_used_when_otlp_headers_unset(self, clean_env):
        # Sanity check the default path: with no OTEL_EXPORTER_OTLP_HEADERS,
        # the DR auth headers are derived from env as before.
        _set_datarobot_env(clean_env)
        cfg = _make_config()
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        headers = mock_exporter.call_args.kwargs["headers"]
        assert headers["X-DataRobot-Api-Key"] == API_KEY
        assert headers["X-DataRobot-Entity-Id"] == ENTITY_ID


class TestNonAsciiSpanAttributes:
    r"""Japanese text must reach the trace viewer readable, not as \uXXXX escapes."""

    @pytest.fixture
    def exporter(self) -> DataRobotOTLPSpanAdapterExporter:
        return DataRobotOTLPSpanAdapterExporter(endpoint=ENDPOINT)

    def test_to_json_string_keeps_japanese_literal(self, exporter):
        data = {"input": "東京の天気は？", "messages": [{"content": "こんにちは"}]}

        result = exporter._to_json_string(data)

        assert "東京の天気は？" in result
        assert "こんにちは" in result
        assert "\\u" not in result
        assert json.loads(result) == data

    def test_to_json_string_ascii_unchanged(self, exporter):
        data = {"input": "hello", "count": 3}

        assert exporter._to_json_string(data) == json.dumps(data)

    def test_to_json_string_keeps_control_char_escapes(self, exporter):
        data = {"input": "日本\x00語"}

        result = exporter._to_json_string(data)

        assert "日本" in result
        assert "\\u0000" in result
        assert json.loads(result) == data

    def test_to_json_string_lone_surrogate_stays_escaped(self, exporter):
        # A lone surrogate can't be UTF-8 encoded for OTLP export; keep NAT's escaped form.
        data = {"input": "日本\ud800"}

        result = exporter._to_json_string(data)

        result.encode("utf-8")
        assert "\\ud800" in result

    @pytest.mark.parametrize(
        "payload",
        [
            {"query": "東京の天気は？"},
            [{"role": "user", "content": "東京の天気は？"}],
            _Payload(query="東京の天気は？"),
        ],
        ids=["dict", "list", "pydantic"],
    )
    def test_serialize_payload_keeps_japanese_literal(self, exporter, payload):
        serialized, is_json = exporter._serialize_payload(payload)

        assert is_json is True
        assert "東京の天気は？" in serialized
        assert "\\u" not in serialized

    def test_serialize_payload_plain_string_untouched(self, exporter):
        assert exporter._serialize_payload("東京") == ("東京", False)


# --- span shrinking: value_obj processor and streamed-output coalescing ---


def _make_step(event_type: IntermediateStepType, step_uuid: str, **data: Any) -> IntermediateStep:
    return IntermediateStep(
        parent_id="root",
        function_ancestry=InvocationNode(
            function_id="root", function_name="root", parent_id=None, parent_name=None
        ),
        payload=IntermediateStepPayload(
            event_type=event_type,
            name="langgraph_agent",
            UUID=step_uuid,
            data=StreamEventData(**data),
        ),
    )


def _streamed_output(n: int) -> list[DRAgentEventResponse]:
    """``n`` per-token chunks of one assistant message, framed by start/end events."""
    chunks = [
        DRAgentEventResponse(
            events=[TextMessageStartEvent(message_id="m1", role="assistant")],
            usage_metrics={"completion_tokens": 0},
        )
    ]
    chunks += [
        DRAgentEventResponse(
            events=[TextMessageContentEvent(message_id="m1", delta=f"tok{i} ")],
            usage_metrics={"completion_tokens": 1},
        )
        for i in range(n)
    ]
    chunks.append(DRAgentEventResponse(events=[TextMessageEndEvent(message_id="m1")]))
    return chunks


class TestDropValueObjProcessor:
    async def test_processor_pops_both_value_obj_keys(self):
        # GIVEN a span carrying NAT's four payload attributes
        span = Span(name="s")
        for key in ("input.value", "input.value_obj", "output.value", "output.value_obj"):
            span.set_attribute(key, "{}")

        # WHEN processed
        result = await DropValueObjAttributesProcessor().process(span)

        # THEN only the value_obj keys are gone and the same span is returned
        assert result is span
        assert set(span.attributes) == {"input.value", "output.value"}

    async def test_processor_tolerates_missing_keys(self):
        # GIVEN a span with no value_obj attributes
        span = Span(name="s", attributes={"nat.event_type": "TOOL_END"})

        # WHEN processed
        await DropValueObjAttributesProcessor().process(span)

        # THEN nothing changes
        assert span.attributes == {"nat.event_type": "TOOL_END"}

    def test_exporter_inserts_processor_at_front_by_default(self):
        # GIVEN an exporter with default options
        exporter = DataRobotOTLPSpanAdapterExporter(endpoint=ENDPOINT)

        # THEN the processor runs first, before NAT's span-to-OTel conversion
        assert isinstance(exporter._processors[0], DropValueObjAttributesProcessor)
        assert exporter._processor_names["drop_value_obj_attributes"] == 0

    def test_exporter_skips_processor_when_disabled(self):
        # GIVEN an exporter with the option off
        exporter = DataRobotOTLPSpanAdapterExporter(
            endpoint=ENDPOINT, drop_value_obj_attributes=False
        )

        # THEN no such processor is in the pipeline
        assert not any(isinstance(p, DropValueObjAttributesProcessor) for p in exporter._processors)


class TestCoalescedEndEvent:
    @pytest.fixture
    def exporter(self) -> DataRobotOTLPSpanAdapterExporter:
        return DataRobotOTLPSpanAdapterExporter(endpoint=ENDPOINT)

    def test_end_event_passes_coalesced_copy_to_nat(self, exporter):
        # GIVEN an END step whose output is a streamed chunk list
        step = _make_step(IntermediateStepType.FUNCTION_END, "u1", output=_streamed_output(3))

        # WHEN the end event is processed
        with patch.object(OTLPSpanAdapterExporter, "_process_end_event") as parent_end:
            exporter._process_end_event(step)

        # THEN NAT receives a copy with one coalesced response and the original is untouched
        received = parent_end.call_args.args[0]
        assert received is not step
        assert received.UUID == step.UUID
        output = received.payload.data.output
        assert isinstance(output, DRAgentEventResponse)
        assert [e.type.value for e in output.events] == [
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TEXT_MESSAGE_END",
        ]
        assert output.events[1].delta == "tok0 tok1 tok2 "
        assert output.usage_metrics == {"completion_tokens": 3}
        assert len(step.payload.data.output) == 5

    @pytest.mark.parametrize(
        "output",
        ["plain text", {"answer": 1}, [], [DRAgentEventResponse(), "not a chunk"], None],
        ids=["str", "dict", "empty-list", "mixed-list", "none"],
    )
    def test_end_event_leaves_other_outputs_untouched(self, exporter, output):
        # GIVEN an END step whose output is not a pure chunk list
        step = _make_step(IntermediateStepType.FUNCTION_END, "u2", output=output)

        # WHEN the end event is processed
        with patch.object(OTLPSpanAdapterExporter, "_process_end_event") as parent_end:
            exporter._process_end_event(step)

        # THEN NAT receives the very same step object
        assert parent_end.call_args.args[0] is step

    def test_end_event_skips_coalescing_when_disabled(self):
        # GIVEN an exporter with coalescing off and a chunk-list output
        exporter = DataRobotOTLPSpanAdapterExporter(
            endpoint=ENDPOINT, coalesce_streamed_output=False
        )
        step = _make_step(IntermediateStepType.FUNCTION_END, "u3", output=_streamed_output(2))

        # WHEN the end event is processed
        with patch.object(OTLPSpanAdapterExporter, "_process_end_event") as parent_end:
            exporter._process_end_event(step)

        # THEN NAT receives the original step
        assert parent_end.call_args.args[0] is step

    def test_end_event_falls_back_to_raw_output_on_coalescer_error(self, exporter):
        # GIVEN a coalescer that blows up
        step = _make_step(IntermediateStepType.FUNCTION_END, "u4", output=_streamed_output(2))

        # WHEN the end event is processed
        with (
            patch(
                "datarobot_genai.dragent.plugins.datarobot_otelcollector."
                "coalesce_dragent_event_responses",
                side_effect=RuntimeError("boom"),
            ),
            patch.object(OTLPSpanAdapterExporter, "_process_end_event") as parent_end,
        ):
            exporter._process_end_event(step)

        # THEN the span is still exported with NAT's raw output
        assert parent_end.call_args.args[0] is step

    def test_end_event_writes_coalesced_output_value_through_nat(self, exporter):
        # GIVEN a started span and an END step with many streamed chunks
        step_uuid = str(uuid.uuid4())
        exporter._process_start_event(
            _make_step(IntermediateStepType.FUNCTION_START, step_uuid, input={"q": "hi"})
        )
        span = exporter._outstanding_spans[step_uuid]
        chunks = _streamed_output(900)
        raw_size = len(json.dumps([c.model_dump(mode="json") for c in chunks]))

        # WHEN the end event runs through NAT's span construction
        # Close the export coroutine NAT hands over, since no loop will run it here.
        with patch.object(exporter, "_create_export_task", side_effect=lambda coro: coro.close()):
            exporter._process_end_event(
                _make_step(IntermediateStepType.FUNCTION_END, step_uuid, output=chunks)
            )

        # THEN output.value holds one response with three events and is far smaller
        output_value = json.loads(span.attributes["output.value"])
        assert len(output_value["events"]) == 3
        assert output_value["events"][1]["delta"].startswith("tok0 tok1 ")
        assert output_value["usage_metrics"] == {"completion_tokens": 900}
        assert len(span.attributes["output.value"]) < raw_size / 10


class TestSpanShrinkingFactoryFlags:
    async def test_flags_default_on(self, clean_env):
        # GIVEN a config with defaults
        _set_datarobot_env(clean_env)
        cfg = _make_config()

        # WHEN the exporter is built
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        # THEN both behaviours are on
        kwargs = mock_exporter.call_args.kwargs
        assert kwargs["drop_value_obj_attributes"] is True
        assert kwargs["coalesce_streamed_output"] is True

    async def test_flags_pass_through_when_disabled(self, clean_env):
        # GIVEN a config that turns both off
        _set_datarobot_env(clean_env)
        cfg = _make_config(drop_value_obj_attributes=False, coalesce_streamed_output=False)

        # WHEN the exporter is built
        with patch(
            "datarobot_genai.dragent.plugins.datarobot_otelcollector.DataRobotOTLPSpanAdapterExporter"
        ) as mock_exporter:
            async with datarobot_otelcollector_telemetry_exporter(cfg, builder=MagicMock()):
                pass

        # THEN both reach the exporter as False
        kwargs = mock_exporter.call_args.kwargs
        assert kwargs["drop_value_obj_attributes"] is False
        assert kwargs["coalesce_streamed_output"] is False
