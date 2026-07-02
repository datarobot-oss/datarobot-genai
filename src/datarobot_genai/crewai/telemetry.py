# Copyright 2025 DataRobot, Inc. and its affiliates.
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

"""Idempotent CrewAI auto-instrumentation for agent telemetry.

The released ``opentelemetry-instrumentation-crewai`` only instruments CrewAI's
synchronous execution path (``kickoff`` -> ``Agent.execute_task`` ->
``Task.execute_sync`` -> ``LLM.call``). CrewAI's native async path
(``akickoff`` -> ``Agent.aexecute_task`` -> ``Task.aexecute_sync`` ->
``LLM.acall``) is uninstrumented, so agents driven via ``akickoff`` emit no
framework spans.

Until async support lands upstream, :class:`DataRobotCrewAIInstrumentor` adds
the async wrappers on top of the released synchronous instrumentor, reusing the
released module's span helpers so span shape stays identical across both paths.

TODO (BUZZOK-31424): remove :class:`DataRobotCrewAIInstrumentor` and the async wrappers once
``opentelemetry-instrumentation-crewai`` ships native ``akickoff``
instrumentation, and revert to instantiating ``CrewAIInstrumentor`` directly.

https://github.com/traceloop/openllmetry/pull/4342
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from collections.abc import Awaitable
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from opentelemetry.instrumentation.crewai import CrewAIInstrumentor
from opentelemetry.instrumentation.crewai.crewai_span_attributes import CrewAISpanAttributes
from opentelemetry.instrumentation.crewai.crewai_span_attributes import set_span_attribute
from opentelemetry.instrumentation.crewai.instrumentation import _create_metrics
from opentelemetry.instrumentation.crewai.instrumentation import _infer_llm_provider_from_model
from opentelemetry.instrumentation.crewai.instrumentation import _record_duration
from opentelemetry.instrumentation.crewai.instrumentation import _set_messages_attributes
from opentelemetry.instrumentation.crewai.instrumentation import _set_response_attributes
from opentelemetry.instrumentation.crewai.instrumentation import is_metrics_enabled
from opentelemetry.instrumentation.crewai.version import __version__
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.metrics import Histogram
from opentelemetry.metrics import get_meter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,  # noqa: N812
)
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import GenAiOperationNameValues
from opentelemetry.semconv_ai import GenAISystem
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.semconv_ai import TraceloopSpanKindValues
from opentelemetry.trace import Span
from opentelemetry.trace import SpanKind
from opentelemetry.trace import Tracer
from opentelemetry.trace import get_tracer
from opentelemetry.trace.status import Status
from opentelemetry.trace.status import StatusCode
from wrapt import wrap_function_wrapper

logger = logging.getLogger(__name__)

_INSTRUMENTED = {"crewai": False}

# Instrumentation scope name; matches the released instrumentor so async spans
# share the same scope as the synchronous ones.
_INSTRUMENTATION_NAME = "opentelemetry.instrumentation.crewai"

_WORKFLOW_SPAN_ATTRIBUTES = {
    GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
    GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
}

# The wrapt callable being wrapped, plus the (instance, args, kwargs) it receives.
_Wrapped = Callable[..., Any]
# The wrapt-style wrapper produced by a factory once a tracer/histograms are bound.
_AsyncWrapper = Callable[[_Wrapped, Any, "tuple[Any, ...]", "dict[str, Any]"], Awaitable[Any]]
# The raw async implementation before @_with_tracer_async binds the tracer/histograms.
_AsyncWrapImpl = Callable[
    [
        Tracer,
        "Histogram | None",
        "Histogram | None",
        _Wrapped,
        Any,
        "tuple[Any, ...]",
        "dict[str, Any]",
    ],
    Awaitable[Any],
]
# The factory that binds tracer/histograms and yields a wrapt wrapper.
_WrapperFactory = Callable[[Tracer, "Histogram | None", "Histogram | None"], _AsyncWrapper]

# Synchronous counterparts of the async wrapper types above.
_SyncWrapper = Callable[[_Wrapped, Any, "tuple[Any, ...]", "dict[str, Any]"], Any]
_SyncWrapImpl = Callable[
    [
        Tracer,
        "Histogram | None",
        "Histogram | None",
        _Wrapped,
        Any,
        "tuple[Any, ...]",
        "dict[str, Any]",
    ],
    Any,
]
_SyncWrapperFactory = Callable[[Tracer, "Histogram | None", "Histogram | None"], _SyncWrapper]


def _with_tracer_async(func: _AsyncWrapImpl) -> _WrapperFactory:
    """Bind tracer/histograms to an async wrapper, mirroring the sync helper upstream."""

    def _factory(
        tracer: Tracer,
        duration_histogram: Histogram | None,
        token_histogram: Histogram | None,
    ) -> _AsyncWrapper:
        async def wrapper(
            wrapped: _Wrapped,
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            return await func(
                tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs
            )

        return wrapper

    return _factory


def _with_tracer_sync(func: _SyncWrapImpl) -> _SyncWrapperFactory:
    """Bind tracer/histograms to a sync wrapper, mirroring :func:`_with_tracer_async`."""

    def _factory(
        tracer: Tracer,
        duration_histogram: Histogram | None,
        token_histogram: Histogram | None,
    ) -> _SyncWrapper:
        def wrapper(
            wrapped: _Wrapped,
            instance: Any,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            return func(
                tracer, duration_histogram, token_histogram, wrapped, instance, args, kwargs
            )

        return wrapper

    return _factory


# ---------------------------------------------------------------------------
# Crew workflow span (akickoff)
# ---------------------------------------------------------------------------


def _finalize_kickoff_span(span: Span, instance: Any, result: Any) -> None:
    if not result:
        return
    class_name = instance.__class__.__name__
    span.set_attribute(f"crewai.{class_name.lower()}.result", str(result))
    span.set_status(Status(StatusCode.OK))
    if class_name == "Crew":
        for attr in ("tasks_output", "token_usage", "usage_metrics"):
            if hasattr(result, attr):
                span.set_attribute(f"crewai.crew.{attr}", str(getattr(result, attr)))


@_with_tracer_async
async def wrap_akickoff(
    tracer: Tracer,
    duration_histogram: Histogram | None,
    token_histogram: Histogram | None,
    wrapped: _Wrapped,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    if getattr(instance, "stream", False):
        # A streaming akickoff returns a lazily async-iterated output; the crew
        # actually runs later (during iteration) via a nested akickoff call with
        # stream=False, which this same wrapper instruments. Creating a span here
        # would only produce an empty one, so defer to that nested call.
        return await wrapped(*args, **kwargs)

    with tracer.start_as_current_span(
        "crewai.workflow",
        kind=SpanKind.INTERNAL,
        attributes=_WORKFLOW_SPAN_ATTRIBUTES,
    ) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = await wrapped(*args, **kwargs)
            _finalize_kickoff_span(span, instance, result)
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


# ---------------------------------------------------------------------------
# Agent span (aexecute_task)
# ---------------------------------------------------------------------------


def _start_agent_span(tracer: Tracer, instance: Any) -> AbstractContextManager[Span]:
    agent_name = instance.role if hasattr(instance, "role") else "agent"
    return tracer.start_as_current_span(
        f"{agent_name}.agent",
        kind=SpanKind.INTERNAL,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.AGENT.value,
            GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
            GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
        },
    )


def _set_agent_request_attributes(span: Span, instance: Any) -> None:
    if hasattr(instance, "role") and instance.role:
        set_span_attribute(span, GenAIAttributes.GEN_AI_AGENT_NAME, instance.role)
    if hasattr(instance, "id"):
        set_span_attribute(span, GenAIAttributes.GEN_AI_AGENT_ID, str(instance.id))


def _finalize_agent_span(span: Span, instance: Any, token_histogram: Histogram | None) -> None:
    summary = instance._token_process.get_summary()
    if token_histogram:
        token_histogram.record(
            summary.prompt_tokens,
            attributes={
                GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
                GenAIAttributes.GEN_AI_TOKEN_TYPE: "input",
                GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.llm.model),
            },
        )
        token_histogram.record(
            summary.completion_tokens,
            attributes={
                GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
                GenAIAttributes.GEN_AI_TOKEN_TYPE: "output",
                GenAIAttributes.GEN_AI_RESPONSE_MODEL: str(instance.llm.model),
            },
        )
    set_span_attribute(span, GenAIAttributes.GEN_AI_REQUEST_MODEL, str(instance.llm.model))
    set_span_attribute(span, GenAIAttributes.GEN_AI_RESPONSE_MODEL, str(instance.llm.model))
    if summary.prompt_tokens:
        set_span_attribute(span, GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS, summary.prompt_tokens)
    if summary.completion_tokens:
        set_span_attribute(
            span, GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS, summary.completion_tokens
        )
    span.set_status(Status(StatusCode.OK))


@_with_tracer_async
async def wrap_aexecute_task(
    tracer: Tracer,
    duration_histogram: Histogram | None,
    token_histogram: Histogram | None,
    wrapped: _Wrapped,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    with _start_agent_span(tracer, instance) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            _set_agent_request_attributes(span, instance)
            result = await wrapped(*args, **kwargs)
            _finalize_agent_span(span, instance, token_histogram)
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


# ---------------------------------------------------------------------------
# Task span (aexecute_sync)
# ---------------------------------------------------------------------------


def _start_task_span(tracer: Tracer, instance: Any) -> AbstractContextManager[Span]:
    task_name = instance.description if hasattr(instance, "description") else "task"
    return tracer.start_as_current_span(
        f"{task_name}.task",
        kind=SpanKind.INTERNAL,
        attributes={
            SpanAttributes.TRACELOOP_SPAN_KIND: TraceloopSpanKindValues.TASK.value,
            GenAIAttributes.GEN_AI_PROVIDER_NAME: GenAISystem.CREWAI.value,
            GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.INVOKE_AGENT.value,
        },
    )


@_with_tracer_async
async def wrap_aexecute_sync(
    tracer: Tracer,
    duration_histogram: Histogram | None,
    token_histogram: Histogram | None,
    wrapped: _Wrapped,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    with _start_task_span(tracer, instance) as span:
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            result = await wrapped(*args, **kwargs)
            set_span_attribute(span, SpanAttributes.TRACELOOP_ENTITY_OUTPUT, str(result))
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


# ---------------------------------------------------------------------------
# LLM span (acall)
# ---------------------------------------------------------------------------


@_with_tracer_async
async def wrap_acall(
    tracer: Tracer,
    duration_histogram: Histogram | None,
    token_histogram: Histogram | None,
    wrapped: _Wrapped,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    model = str(instance.model) if hasattr(instance, "model") else "llm"
    provider = _infer_llm_provider_from_model(getattr(instance, "model", None))
    span_attrs = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: model,
    }
    if provider:
        span_attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = provider

    with tracer.start_as_current_span(
        f"{model}.llm", kind=SpanKind.CLIENT, attributes=span_attrs
    ) as span:
        start_time = time.time()
        try:
            CrewAISpanAttributes(span=span, instance=instance)
            messages_arg = args[0] if args else kwargs.get("messages")
            result = await wrapped(*args, **kwargs)

            _set_messages_attributes(span, messages_arg, result)
            _set_response_attributes(span, instance)
            _record_duration(duration_histogram, start_time, model, provider)

            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


# ---------------------------------------------------------------------------
# DataRobot-specific LLM span (aget_llm_response / get_llm_response)
#
# NOTE: This is NOT part of upstream opentelemetry-instrumentation-crewai and is
# specific to datarobot-genai. The upstream instrumentor wraps
# crewai.llm.LLM.call / LLM.acall, but datarobot-genai ships custom
# crewai.llm.LLM subclasses (LitellmStopWordLLM, RouterLitellmOnlyLLM) that
# override call()/acall() and, on the native tool-calling path, drive litellm
# directly without delegating to super().call()/acall(). The upstream wrap on
# LLM.call/acall therefore never fires for those subclasses. To emit an LLM span
# regardless of the concrete LLM subclass, we instrument CrewAI's LLM-invocation
# choke points crewai.utilities.agent_utils.{aget_llm_response, get_llm_response},
# through which every agent LLM call flows.
# ---------------------------------------------------------------------------


def _extract_llm_and_messages(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """Pull the ``llm`` and ``messages`` positional/keyword args out of a call.

    Matches ``(a)get_llm_response(llm, messages, ...)``.
    """
    llm = args[0] if args else kwargs.get("llm")
    messages = args[1] if len(args) > 1 else kwargs.get("messages")
    return llm, messages


def _llm_span_attributes(llm: Any) -> tuple[str, str | None, dict[str, Any]]:
    """Build the ``{model}.llm`` span name inputs + attributes, mirroring ``wrap_acall``."""
    model = str(llm.model) if hasattr(llm, "model") else "llm"
    provider = _infer_llm_provider_from_model(getattr(llm, "model", None))
    span_attrs: dict[str, Any] = {
        GenAIAttributes.GEN_AI_OPERATION_NAME: GenAiOperationNameValues.CHAT.value,
        GenAIAttributes.GEN_AI_REQUEST_MODEL: model,
    }
    if provider:
        span_attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = provider
    return model, provider, span_attrs


@_with_tracer_async
async def wrap_aget_llm_response(
    tracer: Tracer,
    duration_histogram: Histogram | None,
    token_histogram: Histogram | None,
    wrapped: _Wrapped,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    llm, messages_arg = _extract_llm_and_messages(args, kwargs)
    model, provider, span_attrs = _llm_span_attributes(llm)
    with tracer.start_as_current_span(
        f"{model}.llm", kind=SpanKind.CLIENT, attributes=span_attrs
    ) as span:
        start_time = time.time()
        try:
            if llm is not None:
                CrewAISpanAttributes(span=span, instance=llm)
            result = await wrapped(*args, **kwargs)
            _set_messages_attributes(span, messages_arg, result)
            if llm is not None:
                _set_response_attributes(span, llm)
            _record_duration(duration_histogram, start_time, model, provider)
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


@_with_tracer_sync
def wrap_get_llm_response(
    tracer: Tracer,
    duration_histogram: Histogram | None,
    token_histogram: Histogram | None,
    wrapped: _Wrapped,
    instance: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    llm, messages_arg = _extract_llm_and_messages(args, kwargs)
    model, provider, span_attrs = _llm_span_attributes(llm)
    with tracer.start_as_current_span(
        f"{model}.llm", kind=SpanKind.CLIENT, attributes=span_attrs
    ) as span:
        start_time = time.time()
        try:
            if llm is not None:
                CrewAISpanAttributes(span=span, instance=llm)
            result = wrapped(*args, **kwargs)
            _set_messages_attributes(span, messages_arg, result)
            if llm is not None:
                _set_response_attributes(span, llm)
            _record_duration(duration_histogram, start_time, model, provider)
            span.set_status(Status(StatusCode.OK))
            return result
        except Exception as ex:
            span.set_status(Status(StatusCode.ERROR, str(ex)))
            raise


# ---------------------------------------------------------------------------
# Instrumentor
# ---------------------------------------------------------------------------

_ASYNC_WRAP_TARGETS = (
    ("crewai.crew", "Crew.akickoff", wrap_akickoff),
    ("crewai.agent", "Agent.aexecute_task", wrap_aexecute_task),
    ("crewai.task", "Task.aexecute_sync", wrap_aexecute_sync),
    ("crewai.llm", "LLM.acall", wrap_acall),
)

_ASYNC_UNWRAP_TARGETS = (
    ("crewai.crew.Crew", "akickoff"),
    ("crewai.agent.Agent", "aexecute_task"),
    ("crewai.task.Task", "aexecute_sync"),
    ("crewai.llm.LLM", "acall"),
)

# DataRobot-specific (NOT upstream) LLM choke-point wrappers. See the note above
# the wrap_*get_llm_response definitions: these capture an LLM span for
# datarobot-genai's custom crewai.llm.LLM subclasses that bypass LLM.call/acall.
#
# CrewAI's agent executors import these choke points *by name* at module load
# (``from crewai.utilities.agent_utils import get_llm_response``), so each
# executor holds its own reference. Patching only the source module
# (``crewai.utilities.agent_utils``) therefore never intercepts the executor's
# calls -- the bare ``get_llm_response(...)`` / ``await aget_llm_response(...)``
# resolves to the executor's own (unwrapped) global. We additionally wrap the
# names in every executor module that imports them so the ``{model}.llm`` span
# is emitted on both the sync and async agent paths. ``_instrument`` skips any
# target that is already wrapped to avoid double-wrapping (wrapt may import an
# executor module while wrapping, at which point its by-name import copies an
# already-wrapped reference from the source module).
_DATAROBOT_WRAP_TARGETS = (
    ("crewai.utilities.agent_utils", "aget_llm_response", wrap_aget_llm_response),
    ("crewai.utilities.agent_utils", "get_llm_response", wrap_get_llm_response),
    ("crewai.agents.crew_agent_executor", "aget_llm_response", wrap_aget_llm_response),
    ("crewai.agents.crew_agent_executor", "get_llm_response", wrap_get_llm_response),
    ("crewai.lite_agent", "get_llm_response", wrap_get_llm_response),
    ("crewai.experimental.agent_executor", "get_llm_response", wrap_get_llm_response),
)

_DATAROBOT_UNWRAP_TARGETS = (
    ("crewai.utilities.agent_utils", "aget_llm_response"),
    ("crewai.utilities.agent_utils", "get_llm_response"),
    ("crewai.agents.crew_agent_executor", "aget_llm_response"),
    ("crewai.agents.crew_agent_executor", "get_llm_response"),
    ("crewai.lite_agent", "get_llm_response"),
    ("crewai.experimental.agent_executor", "get_llm_response"),
)


def _is_already_wrapped(module_name: str, dotted_name: str) -> bool:
    """Return whether ``module_name.dotted_name`` is already a wrapt wrapper.

    Used to keep :meth:`DataRobotCrewAIInstrumentor._instrument` idempotent when
    a choke point is wrapped both at its source module and in the executor
    modules that import it by name: wrapt imports an executor module while
    wrapping it, and its ``from ... import`` may copy an already-wrapped
    reference from the source module. Raises ``ModuleNotFoundError`` /
    ``AttributeError`` (propagated to the caller's ``except``) when the target
    does not exist on the installed CrewAI version.
    """
    obj: Any = importlib.import_module(module_name)
    for part in dotted_name.split("."):
        obj = getattr(obj, part)
    return hasattr(obj, "__wrapped__")


class DataRobotCrewAIInstrumentor(CrewAIInstrumentor):
    """CrewAIInstrumentor extended with async execution-path (akickoff) wrappers.

    The synchronous wrapping is delegated to the released base class; this
    subclass adds wrappers for CrewAI's native async methods. Each async method
    is wrapped defensively so instrumentation still succeeds on CrewAI releases
    that predate a given method.
    """

    def _instrument(self, **kwargs: Any) -> None:
        super()._instrument(**kwargs)

        tracer = get_tracer(_INSTRUMENTATION_NAME, __version__, kwargs.get("tracer_provider"))
        meter = get_meter(_INSTRUMENTATION_NAME, __version__, kwargs.get("meter_provider"))
        token_histogram = None
        duration_histogram = None
        if is_metrics_enabled():
            token_histogram, duration_histogram = _create_metrics(meter)

        # Upstream async wrappers plus the DataRobot-specific LLM choke-point
        # wrappers (see _DATAROBOT_WRAP_TARGETS note above).
        wrap_targets: tuple[tuple[str, str, Any], ...] = (
            *_ASYNC_WRAP_TARGETS,
            *_DATAROBOT_WRAP_TARGETS,
        )
        for module, method, factory in wrap_targets:
            try:
                if _is_already_wrapped(module, method):
                    logger.debug("CrewAI method %s.%s already wrapped; skipping", module, method)
                    continue
                wrap_function_wrapper(
                    module, method, factory(tracer, duration_histogram, token_histogram)
                )
            except (AttributeError, ModuleNotFoundError):
                logger.debug("CrewAI method %s.%s not found; skipping", module, method)

    def _uninstrument(self, **kwargs: Any) -> None:
        super()._uninstrument(**kwargs)
        for module, method in (*_ASYNC_UNWRAP_TARGETS, *_DATAROBOT_UNWRAP_TARGETS):
            try:
                unwrap(module, method)
            except (AttributeError, ModuleNotFoundError):
                pass


def instrument() -> None:
    """Idempotently enable CrewAI instrumentation, including the async path."""
    if _INSTRUMENTED["crewai"]:
        logger.info("CrewAI instrumentation already enabled")
        return
    try:
        DataRobotCrewAIInstrumentor().instrument()
        os.environ.setdefault("CREWAI_TESTING", "true")
        _INSTRUMENTED["crewai"] = True
    except Exception as e:
        logger.info(f"CrewAI instrumentation failed: {e}")
