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

"""Compatibility shims for third-party OpenTelemetry instrumentation.

The upstream fix for cross-context OTel detach errors landed in
``llama-index-observability-otel`` (https://github.com/run-llama/llama_index/pull/21587,
released in >= 0.6.2). We still need a local shim because ``instrument(framework="llamaindex")``
uses OpenLLMetry's ``opentelemetry-instrumentation-llamaindex`` instead — a separate span
handler on the same LlamaIndex dispatcher that has not picked up that fix. Drop this module
when we migrate to ``llama-index-observability-otel`` or OpenLLMetry ships an equivalent
``SpanHolder.end`` change.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Mutable module state to avoid global reassignment (see agent.py instrumentation state).
_STATE: dict[str, Any] = {
    "patched": False,
    "original_new_span": None,
    "original_end": None,
}


def patch_llamaindex_otel_context_detach() -> None:
    """Fix cross-async-context OTel detach errors from LlamaIndex workflows.

    LlamaIndex AgentWorkflow steps can end OpenLLMetry spans from a copied
    ``contextvars`` continuation. The stored detach token belongs to the
    original context, so ``ContextVar.reset(token)`` raises ``ValueError``.

    ``opentelemetry-instrumentation-llamaindex`` ends spans via
    ``opentelemetry.context.detach``, which logs ``Failed to detach context``
    at ERROR without re-raising. This patches the OpenLLMetry ``SpanHolder``
    to reset the token directly and, on cross-context failure, re-attach the
    saved previous OTel context when the ended span is still current.

    The same logic was merged upstream for LlamaIndex's native OTel integration
    in https://github.com/run-llama/llama_index/pull/21587 (``llama-index-observability-otel``
    >= 0.6.2), but that package is not what ``instrument(framework="llamaindex")`` activates.
    """
    if _STATE["patched"]:
        return

    try:
        from opentelemetry.instrumentation.llamaindex import dispatcher_wrapper as dw
    except ImportError:
        return

    _STATE["original_new_span"] = dw.OpenLLMetrySpanHandler.new_span
    _STATE["original_end"] = dw.SpanHolder.end

    def patched_new_span(
        self: Any,
        id_: str,
        bound_args: Any,
        instance: Any | None = None,
        parent_span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        from opentelemetry import context as context_api

        previous_context = context_api.get_current()
        span_holder = _STATE["original_new_span"](
            self,
            id_,
            bound_args,
            instance=instance,
            parent_span_id=parent_span_id,
            tags=tags,
            **kwargs,
        )
        if span_holder is not None and span_holder.token is not None:
            span_holder._previous_context = previous_context
        return span_holder

    def patched_end(self: Any, should_detach_context: bool = True) -> None:
        from opentelemetry import context as context_api
        from opentelemetry.trace import get_current_span

        if not self._active:
            return

        self._active = False
        if self.otel_span:
            self.otel_span.end()
        if self.token and should_detach_context:
            previous_context = getattr(self, "_previous_context", None)
            try:
                self.token.var.reset(self.token)
            except ValueError as err:
                if "different Context" not in str(err):
                    logger.warning(
                        "Error detaching OTel context for span %s",
                        self.span_id,
                        exc_info=True,
                    )
                elif (
                    previous_context is not None
                    and self.otel_span is not None
                    and get_current_span() is self.otel_span
                ):
                    context_api.attach(previous_context)
            except Exception:
                logger.warning(
                    "Error detaching OTel context for span %s",
                    self.span_id,
                    exc_info=True,
                )

    dw.OpenLLMetrySpanHandler.new_span = patched_new_span  # type: ignore[method-assign]
    dw.SpanHolder.end = patched_end  # type: ignore[method-assign]
    _STATE["patched"] = True
