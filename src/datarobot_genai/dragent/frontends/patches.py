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

"""Compatibility patches for nvidia-nat-crewai 1.4.1 with crewai >= 1.1.0.

nvidia-nat-crewai 1.4.1 expects crewai < 1.0.0 response format where
``choice.model_extra["message"]`` holds the LLM message. In crewai >= 1.1.0,
``message`` is a proper attribute on the choice object. This module patches
the callback handler to support both formats.

When ``stream=True``, LiteLLM returns a stream object (e.g. ``CustomStreamWrapper``)
without ``.choices``. NAT's ``wrapped_llm_call`` only handles non-streaming
``ModelResponse`` objects, so streaming calls must bypass that wrapper and call
``litellm.completion`` directly.

TODO(BUZZOK-29844): Remove once nvidia-nat-crewai ships a fix upstream.
Upstream issue: https://github.com/NVIDIA/NeMo-Agent-Toolkit/issues/1802
"""

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def patch_crewai_callback_handler() -> None:
    """Patch CrewAIProfilerHandler._llm_call_monkey_patch for crewai >= 1.1.0 compatibility."""
    try:
        from nat.plugins.crewai.crewai_callback_handler import CrewAIProfilerHandler
    except ImportError:
        return

    if getattr(CrewAIProfilerHandler._llm_call_monkey_patch, "_dr_patched", False):
        return

    _original_method = CrewAIProfilerHandler._llm_call_monkey_patch

    def _patched_llm_call_monkey_patch(self: Any) -> Callable[..., Any]:
        """Wrap the original monkey patch to inject message into model_extra before it runs."""
        original_litellm = self._original_llm_call

        def fixed_wrapped(*args: Any, **kwargs: Any) -> Any:
            def compat_completion(*a: Any, **kw: Any) -> Any:
                output = original_litellm(*a, **kw)
                choices = getattr(output, "choices", None)
                if not choices:
                    return output
                for choice in choices:
                    if (
                        choice.model_extra is not None
                        and "message" not in choice.model_extra
                        and hasattr(choice, "message")
                        and choice.message is not None
                    ):
                        choice.model_extra["message"] = choice.message.model_dump()
                return output

            self._original_llm_call = compat_completion
            try:
                patched_wrapped = _original_method(self)

                def dispatch(*a: Any, **kw: Any) -> Any:
                    # NAT's wrapped_llm_call reads output.choices; streaming returns an iterator.
                    if kw.get("stream"):
                        return original_litellm(*a, **kw)
                    return patched_wrapped(*a, **kw)

                return dispatch(*args, **kwargs)
            finally:
                self._original_llm_call = original_litellm

        return fixed_wrapped

    CrewAIProfilerHandler._llm_call_monkey_patch = _patched_llm_call_monkey_patch
    CrewAIProfilerHandler._llm_call_monkey_patch._dr_patched = True
    logger.debug("Patched CrewAIProfilerHandler for crewai >= 1.1.0 compatibility")


def patch_generate_streaming_response() -> None:
    """Ensure NAT streaming endpoints end the SSE connection when the agent errors.

    NAT's ``generate_streaming_response`` runs result production in ``asyncio.create_task``.
    If ``runner.result_stream()`` raises, the producer never calls ``queue.close()``,
    so consumers block forever on ``async for item in q``. We always close the queue
    in a ``finally`` block and await the producer task so failures propagate.
    """
    try:
        import nat.front_ends.fastapi.response_helpers as rh
    except ImportError:
        return

    if getattr(rh.generate_streaming_response, "_dr_patched", False):
        return

    import asyncio
    import typing

    from nat.data_models.api_server import ResponseIntermediateStep
    from nat.data_models.api_server import ResponsePayloadOutput
    from nat.data_models.api_server import ResponseSerializable
    from nat.data_models.step_adaptor import StepAdaptorConfig
    from nat.front_ends.fastapi.intermediate_steps_subscriber import pull_intermediate
    from nat.front_ends.fastapi.step_adaptor import StepAdaptor
    from nat.utils.producer_consumer_queue import AsyncIOProducerConsumerQueue

    async def generate_streaming_response(  # noqa: PLR0915
        payload: typing.Any,
        *,
        session: typing.Any,
        streaming: bool,
        step_adaptor: StepAdaptor = StepAdaptor(StepAdaptorConfig()),
        result_type: type | None = None,
        output_type: type | None = None,
    ) -> typing.AsyncGenerator[typing.Any, None]:
        async with session.run(payload) as runner:
            q: AsyncIOProducerConsumerQueue = AsyncIOProducerConsumerQueue()
            intermediate_complete = await pull_intermediate(q, step_adaptor)

            async def pull_result() -> None:
                try:
                    if session.workflow.has_streaming_output and streaming:
                        async for chunk in runner.result_stream(to_type=output_type):
                            await q.put(chunk)
                    else:
                        result = await runner.result(to_type=result_type)
                        await q.put(runner.convert(result, output_type))

                    await intermediate_complete.wait()
                finally:
                    await q.close()

            task = asyncio.create_task(pull_result())
            try:
                async for item in q:
                    if isinstance(item, ResponseSerializable):
                        yield item
                    else:
                        yield ResponsePayloadOutput(payload=item)
            finally:
                await q.close()
                await task

    async def generate_streaming_response_full(  # noqa: PLR0915
        payload: typing.Any,
        *,
        session: typing.Any,
        streaming: bool,
        result_type: type | None = None,
        output_type: type | None = None,
        filter_steps: str | None = None,
    ) -> typing.AsyncGenerator[typing.Any, None]:
        allowed_types: set[str] | None = None
        if filter_steps:
            if filter_steps.lower() == "none":
                allowed_types = set()
            else:
                allowed_types = set(filter_steps.split(","))

        async with session.run(payload) as runner:
            q: AsyncIOProducerConsumerQueue = AsyncIOProducerConsumerQueue()
            intermediate_complete = await pull_intermediate(q, None)

            async def pull_result() -> None:
                try:
                    if session.workflow.has_streaming_output and streaming:
                        async for chunk in runner.result_stream(to_type=output_type):
                            await q.put(chunk)
                    else:
                        result = await runner.result(to_type=result_type)
                        await q.put(runner.convert(result, output_type))

                    await intermediate_complete.wait()
                finally:
                    await q.close()

            task = asyncio.create_task(pull_result())
            try:
                async for item in q:
                    if isinstance(item, ResponseIntermediateStep):
                        if allowed_types is None or item.type in allowed_types:
                            yield item
                    else:
                        yield ResponsePayloadOutput(payload=item)
            finally:
                await q.close()
                await task

    rh.generate_streaming_response = generate_streaming_response
    rh.generate_streaming_response_full = generate_streaming_response_full
    rh.generate_streaming_response._dr_patched = True
    rh.generate_streaming_response_full._dr_patched = True
    logger.debug("Patched NAT generate_streaming_response for producer failure handling")
