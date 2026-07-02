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

"""Minimal subprocess runner mirroring ``datarobot-user-models``/``run_agent.py``.

Used by ``test_run_agent_inline.py`` to exercise ``execute_dragent_inline``
end-to-end as if it were running inside the agent custom-model environment:
chat completion params come in as JSON on the command line and the resulting
``ChatCompletion`` is written to disk as JSON for the test to read back.

CLI shape intentionally matches the relevant subset of ``run_agent.py``:
``--chat_completion``, ``--custom_model_dir``, ``--output_path``. An optional
``--config_file`` overrides the default ``workflow.yaml`` lookup so tracing
variants of the same agent (e.g. ``workflow-tracing.yaml``) can be exercised.

When executed as ``__main__``, the runner wraps ``main()`` in a short
ipykernel-like host (sync code on a thread that already has a running asyncio
loop) so subprocess e2e tests match production agentic notebooks
(``datarobot/notebooks`` → ``ipykernel``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from datarobot_genai.core.telemetry.datarobot_otel import bootstrap_otel_provider_for_datarobot
from datarobot_genai.dragent import execute_dragent_inline

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _run_sync_under_ipykernel_like_loop(fn: Callable[[], _T]) -> _T:
    """Run *fn* the way ``ipykernel`` runs a synchronous notebook cell.

    The notebooks service sends an ``execute_request`` to ``ipykernel``; user
    code runs on the kernel thread while that thread's asyncio loop is already
    active. Wrapping *fn* in ``loop.run_until_complete`` models that layout for
    subprocess-based e2e runners that do not start from a live Jupyter kernel.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:

        async def _execute_sync_cell() -> _T:
            return fn()

        return loop.run_until_complete(_execute_sync_cell())
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chat_completion",
        required=True,
        help="OpenAI ChatCompletion create params as a JSON string.",
    )
    parser.add_argument(
        "--custom_model_dir",
        required=True,
        help="Directory holding the agent's workflow.yaml (and code).",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="File to write the result JSON to (matches run_agent.py's --output_path).",
    )
    parser.add_argument(
        "--config_file",
        required=False,
        default=None,
        help=(
            "Optional path to a workflow YAML; defaults to "
            "<custom_model_dir>/workflow.yaml when omitted."
        ),
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()
    chat_completion = json.loads(args.chat_completion)
    custom_model_dir = Path(args.custom_model_dir)
    output_path = Path(args.output_path)

    config_file = Path(args.config_file) if args.config_file else None
    logger.info("Executing dragent inline from %s (config_file=%s)", custom_model_dir, config_file)

    bootstrap_otel_provider_for_datarobot()

    result = execute_dragent_inline(
        chat_completion=chat_completion,
        custom_model_dir=custom_model_dir,
        config_file=config_file,
    )

    output_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2))
    logger.info("Wrote result to %s", output_path)
    _flush_otel_tracer_provider()
    return 0


def _flush_otel_tracer_provider() -> None:
    """Force-flush the global OTel SDK TracerProvider so spans land before exit.

    NAT's own exporter pipeline flushes via its async lifecycle; the
    bootstrap-installed ``BatchSpanProcessor`` (from
    ``datarobot_genai.core.telemetry.datarobot_otel``) does not get torn down here, so we
    call ``shutdown()`` explicitly. Wrapped in a broad except because telemetry
    setup must never break the subprocess exit path.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.shutdown()
    except Exception:  # pragma: no cover — diagnostic-only path
        logger.exception("Failed to flush OTel TracerProvider before exit")


if __name__ == "__main__":
    try:
        exit_code = _run_sync_under_ipykernel_like_loop(main)
    finally:
        # Flush log handlers and stdio buffers before the subprocess exits so
        # the parent test runner sees the full stdout/stderr captured output
        # (including any tracebacks emitted from inside ``main()``).
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
    sys.exit(exit_code)
