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

"""Stop the agent run when a streaming client goes away.

NAT's ``generate_streaming_response`` runs the workflow in a separate producer task and
hands its output to the HTTP response through a queue. When the client disconnects,
Starlette cancels the response generator, but that generator's ``finally`` only closes
the queue. The producer task is never cancelled. It keeps calling the LLM and tools until
its next ``q.put`` raises ``QueueClosed``, which can be one slow LLM call (tens of seconds)
or more later.

While it lingers, the abandoned run overlaps the next request in the same worker. That
overlap is what breaks:

* trace isolation: the abandoned run's spans finish late, outside its request, and land
  in the next request's export window (often with a parentless guard span);
* NAT's per-run span stack ("Current span ID stack is not equal to the previous stack");
* any framework state that is process-global rather than per-request (e.g. CrewAI's
  event bus scope).

:func:`patch_nat_streaming_disconnect_cancellation` swaps in a copy of NAT's function that
cancels the producer when the consumer exits early, and waits briefly (shielded from the
caller's cancellation) for it to unwind before the runner context closes.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
from collections.abc import AsyncGenerator
from typing import Any

import anyio
from nat.data_models.api_server import ResponsePayloadOutput
from nat.data_models.api_server import ResponseSerializable
from nat.data_models.step_adaptor import StepAdaptorConfig
from nat.front_ends.fastapi import response_helpers
from nat.front_ends.fastapi.intermediate_steps_subscriber import pull_intermediate
from nat.front_ends.fastapi.step_adaptor import StepAdaptor
from nat.utils.producer_consumer_queue import AsyncIOProducerConsumerQueue

logger = logging.getLogger(__name__)

# How long a cancelled producer gets to unwind (close spans, release its LLM/MCP calls)
# before the runner context exits. Cancellation normally lands within milliseconds; this
# only bounds a producer that swallows or delays it.
PRODUCER_CANCEL_GRACE_SECONDS = 5.0

# Parameters of the NAT function this module replaces. The replacement is a copy of NAT's
# body, so if NAT changes the signature the copy is stale: skip patching and warn rather
# than silently changing streaming behavior.
_EXPECTED_PARAMETERS = (
    "payload",
    "session",
    "streaming",
    "step_adaptor",
    "result_type",
    "output_type",
)

# Modules that bind ``generate_streaming_response`` at import time and so need their own
# attribute replaced. ``response_helpers`` covers the /generate and /chat HTTP routes (they
# reach it through ``generate_streaming_response_as_str``); the others are the websocket
# and interactive-execution paths.
_PATCHED_MODULES = (
    "nat.front_ends.fastapi.response_helpers",
    "nat.front_ends.fastapi.message_handler",
    "nat.front_ends.fastapi.http_interactive_runner",
)

_PATCH_STATE: dict[str, bool] = {"patched": False}

# NAT's own function, captured before any patch so the signature guard stays checkable.
NAT_GENERATE_STREAMING_RESPONSE = response_helpers.generate_streaming_response


def _log_producer_exception(task: asyncio.Task[Any]) -> None:
    """Retrieve a cancelled-away producer's exception so asyncio doesn't warn about it."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.debug("Streaming producer ended with an error after client exit", exc_info=exc)


async def _stop_producer(task: asyncio.Task[Any] | None) -> None:
    """Cancel *task* if it is still running and give it a bounded window to unwind."""
    if task is None or task.done():
        return
    task.cancel()
    task.add_done_callback(_log_producer_exception)
    with anyio.move_on_after(PRODUCER_CANCEL_GRACE_SECONDS):
        await asyncio.wait({task})
    if not task.done():
        logger.warning(
            "Streaming producer did not stop within %ss of the client leaving",
            PRODUCER_CANCEL_GRACE_SECONDS,
        )


async def generate_streaming_response(
    payload: Any,
    *,
    session: Any,
    streaming: bool,
    step_adaptor: Any = None,
    result_type: type | None = None,
    output_type: type | None = None,
) -> AsyncGenerator[Any]:
    """Copy of NAT 1.7's ``generate_streaming_response`` that cancels its producer on exit.

    Only the ``finally`` differs from NAT: when the consumer stops early (client disconnect,
    cancellation, ``aclose()``), the producer task is cancelled and awaited instead of being
    left to run. The wait is shielded because the caller's cancel scope is already cancelled
    and would otherwise interrupt it at once.
    """
    if step_adaptor is None:
        step_adaptor = StepAdaptor(StepAdaptorConfig())

    async with session.run(payload) as runner:
        q: AsyncIOProducerConsumerQueue[Any] = AsyncIOProducerConsumerQueue()

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

        task: asyncio.Task[None] | None = None
        try:
            task = asyncio.create_task(pull_result())

            async for item in q:
                if isinstance(item, ResponseSerializable):
                    yield item
                else:
                    yield ResponsePayloadOutput(payload=item)

            # Re-raise any exception from the producer so callers can handle it
            await task
        finally:
            with anyio.CancelScope(shield=True):
                await _stop_producer(task)
                await q.close()


def patch_nat_streaming_disconnect_cancellation() -> None:
    """Make NAT's streaming responses cancel the workflow when the client goes away.

    Idempotent. Skips (with a warning) when NAT's function no longer has the signature the
    copy above was written against.
    """
    if _PATCH_STATE["patched"]:
        return

    original = response_helpers.generate_streaming_response
    parameters = tuple(inspect.signature(original).parameters)
    if parameters != _EXPECTED_PARAMETERS:
        logger.warning(
            "NAT generate_streaming_response signature changed to %s; client disconnects "
            "will not cancel the agent run. Update %s for this NAT version.",
            parameters,
            __name__,
        )
        return

    for module_name in _PATCHED_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            logger.debug("Could not import %s for stream cancellation patch", module_name)
            continue
        if getattr(module, "generate_streaming_response", None) is original:
            module.generate_streaming_response = generate_streaming_response  # type: ignore[attr-defined]

    _PATCH_STATE["patched"] = True
    logger.debug("Patched NAT generate_streaming_response to cancel on client disconnect")
