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

import logging
from contextvars import copy_context
from unittest.mock import MagicMock
from unittest.mock import patch

import opentelemetry.context as otel_context

from datarobot_genai.core.telemetry.llamaindex_otel_compat import (
    patch_llamaindex_otel_context_detach,
)


class TestLlamaIndexOtelCompat:
    def test_patch_swallows_cross_context_detach_error(self, caplog):
        patch_llamaindex_otel_context_detach()

        mock_token = MagicMock()
        mock_token.var.reset.side_effect = ValueError("token was created in a different Context")
        span_holder = MagicMock(
            _active=True,
            span_id="span-1",
            otel_span=MagicMock(),
            token=mock_token,
            _previous_context=None,
        )

        dispatcher_wrapper = __import__(
            "opentelemetry.instrumentation.llamaindex.dispatcher_wrapper",
            fromlist=["SpanHolder"],
        )

        with caplog.at_level(logging.ERROR, logger="opentelemetry.context"):
            dispatcher_wrapper.SpanHolder.end(span_holder)

        assert "Failed to detach context" not in caplog.text
        mock_token.var.reset.assert_called_once_with(mock_token)

    def test_patch_logs_unexpected_detach_errors(self, caplog):
        patch_llamaindex_otel_context_detach()

        mock_token = MagicMock()
        mock_token.var.reset.side_effect = RuntimeError("unexpected detach failure")
        span_holder = MagicMock(
            _active=True,
            span_id="span-1",
            otel_span=MagicMock(),
            token=mock_token,
            _previous_context=None,
        )

        dispatcher_wrapper = __import__(
            "opentelemetry.instrumentation.llamaindex.dispatcher_wrapper",
            fromlist=["SpanHolder"],
        )

        with caplog.at_level(logging.WARNING):
            dispatcher_wrapper.SpanHolder.end(span_holder)

        assert "Error detaching OTel context for span span-1" in caplog.text

    def test_patch_is_idempotent(self):
        patch_llamaindex_otel_context_detach()

        dispatcher_wrapper = __import__(
            "opentelemetry.instrumentation.llamaindex.dispatcher_wrapper",
            fromlist=["SpanHolder"],
        )
        first_end = dispatcher_wrapper.SpanHolder.end
        patch_llamaindex_otel_context_detach()
        assert dispatcher_wrapper.SpanHolder.end is first_end

    def test_span_exit_from_copied_context_does_not_log_detach_error(self, caplog):
        patch_llamaindex_otel_context_detach()

        dispatcher_wrapper = __import__(
            "opentelemetry.instrumentation.llamaindex.dispatcher_wrapper",
            fromlist=["OpenLLMetrySpanHandler", "SpanHolder"],
        )
        tracer = MagicMock()
        handler = dispatcher_wrapper.OpenLLMetrySpanHandler(tracer)
        bound_args = MagicMock()
        bound_args.arguments = {}
        span_id = "SentenceSplitter.split_text_metadata_aware-a2f2a780-2fa6-4682-a88e-80dc1f1ebe6a"

        clean_token = otel_context.attach(otel_context.Context())
        try:
            span_holder = handler.new_span(id_=span_id, bound_args=bound_args)
            assert span_holder is not None

            def exit_from_copied_context() -> None:
                caplog.set_level(logging.ERROR, logger="opentelemetry.context")
                dispatcher_wrapper.SpanHolder.end(span_holder)

            copy_context().run(exit_from_copied_context)
            assert "Failed to detach context" not in caplog.text
            assert span_holder._active is False
        finally:
            otel_context.detach(clean_token)

    def test_span_exit_reattaches_previous_context_on_cross_context_failure(self):
        patch_llamaindex_otel_context_detach()

        dispatcher_wrapper = __import__(
            "opentelemetry.instrumentation.llamaindex.dispatcher_wrapper",
            fromlist=["SpanHolder"],
        )
        previous_context = otel_context.Context()
        mock_span = MagicMock()
        mock_token = MagicMock()
        mock_token.var.reset.side_effect = ValueError("token was created in a different Context")
        span_holder = MagicMock(
            _active=True,
            span_id="span-1",
            otel_span=mock_span,
            token=mock_token,
            _previous_context=previous_context,
        )

        with (
            patch("opentelemetry.trace.get_current_span", return_value=mock_span),
            patch("opentelemetry.context.attach") as mock_attach,
        ):
            dispatcher_wrapper.SpanHolder.end(span_holder)

        mock_attach.assert_called_once_with(previous_context)
