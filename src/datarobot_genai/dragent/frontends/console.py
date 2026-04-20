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

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click
from colorama import Fore
from colorama import Style
from nat.builder.context import Context
from nat.data_models.step_adaptor import StepAdaptorConfig
from nat.front_ends.console.console_front_end_config import ConsoleFrontEndConfig
from nat.front_ends.console.console_front_end_plugin import ConsoleFrontEndPlugin
from nat.front_ends.console.console_front_end_plugin import prompt_for_input_cli
from nat.runtime.session import SessionManager
from pydantic import Field

from datarobot_genai.dragent.cli.render import render_object_event

from .step_adaptor import DRAgentNestedReasoningStepAdaptor

logger = logging.getLogger(__name__)


# Registered as "dragent_console" so it doesn't conflict with NAT's built-in "console" frontend.
# Used by `nat dragent run`.
class DRAgentConsoleFrontEndConfig(ConsoleFrontEndConfig, name="dragent_console"):  # type: ignore
    """Frontend config for running a dragent workflow from the console.

    Overrides CLI flag names to match the cli.py interface so the Taskfile can
    pass CLI_ARGS straight through without flag translation.
    """

    model_config = {"populate_by_name": True}

    input_query: list[str] | None = Field(
        default=None,
        alias="user_prompt",
        description="User prompt string to send to the workflow.",
    )
    input_file: Path | None = Field(
        default=None,
        alias="completion_json",
        description="Path to a JSON file containing the chat completion payload.",
    )


class DRAgentConsoleFrontEndPlugin(ConsoleFrontEndPlugin):
    """Console frontend plugin with DRAgent step adaptor for intermediate step visibility.

    Why we override ``run_workflow()`` instead of a smaller hook:

    NAT's console ``run_workflow()`` collects the final result via ``runner.result()``
    but never subscribes to intermediate steps. The event stream (a ContextVar-backed
    Subject) is only available inside ``session.run()``. So we override ``run_workflow()``
    to subscribe to ``Context.intermediate_step_manager`` inside the ``session.run()``
    block, routing events through ``DRAgentNestedReasoningStepAdaptor`` to print
    reasoning/tool/text events to stderr.

    If NAT adds a step adaptor hook to the console base class in the future, this
    override should be replaced with that hook.
    """

    def _get_step_adaptor(self) -> DRAgentNestedReasoningStepAdaptor:
        return DRAgentNestedReasoningStepAdaptor(StepAdaptorConfig())

    @staticmethod
    def _subscribe_intermediate_steps(
        step_adaptor: DRAgentNestedReasoningStepAdaptor,
        streamed_output: list[bool],
    ) -> None:
        """Subscribe to NAT's intermediate step stream and print AG-UI events to the terminal.

        Must be called inside ``session.run()`` where the ContextVar-backed event
        stream Subject is active.  Uses the same ``Context.intermediate_step_manager``
        observable that the FastAPI path uses via ``pull_intermediate``.

        *streamed_output* is a single-element list used as a mutable flag; it is set
        to ``[True]`` when at least one text delta has been printed, so callers can
        skip the redundant final-result dump.
        """
        context = Context.get()

        def on_next(step: object) -> None:
            result = step_adaptor.process(step)  # type: ignore[arg-type]
            if result is None:
                return
            for event in getattr(result, "events", []):
                if render_object_event(event):
                    streamed_output[0] = True

        def on_error(exc: Exception) -> None:
            logger.debug("Intermediate step stream error: %s", exc)

        def on_complete() -> None:
            logger.debug("Intermediate step stream completed")

        context.intermediate_step_manager.subscribe(
            on_next=on_next,
            on_error=on_error,
            on_complete=on_complete,
        )

    async def run_workflow(self, session_manager: SessionManager) -> None:
        # See class docstring for why this override is necessary.
        runner_outputs = None
        streamed_output: list[bool] = [False]

        # --- BEGIN COPY from nat.front_ends.console.console_front_end_plugin (nvidia-nat 1.6.0) ---
        # Only change: self._subscribe_intermediate_steps() injected inside
        # each session.run() block. Output formatting delegated to super().
        if self.front_end_config.input_query is not None:

            async def run_single_query(query: str) -> str:
                async with session_manager.session(
                    user_id=self.front_end_config.user_id,
                    user_input_callback=prompt_for_input_cli,
                    user_authentication_callback=self.auth_flow_handler.authenticate,
                ) as session:
                    async with session.run(query) as runner:
                        # Each query gets its own adaptor to avoid shared mutable state
                        # (function_level, seen_llm_new_token) across concurrent coroutines.
                        self._subscribe_intermediate_steps(
                            self._get_step_adaptor(), streamed_output
                        )
                        return await runner.result(to_type=str)

            input_list = list(self.front_end_config.input_query)
            runner_outputs = await asyncio.gather(
                *[run_single_query(query) for query in input_list],
                return_exceptions=False,
            )

        elif self.front_end_config.input_file is not None:
            with open(self.front_end_config.input_file, encoding="utf-8") as f:
                input_content = f.read()
            async with session_manager.session(user_id=self.front_end_config.user_id) as session:
                async with session.run(input_content) as runner:
                    self._subscribe_intermediate_steps(self._get_step_adaptor(), streamed_output)
                    runner_outputs = await runner.result(to_type=str)
        else:
            raise RuntimeError("No input provided. Should have been caught by pre_run.")
        # --- END COPY from nat.front_ends.console.console_front_end_plugin (nvidia-nat 1.6.0) ---

        if streamed_output[0]:
            click.echo(f"\n{Fore.GREEN}\u2705 Run finished.{Style.RESET_ALL}", err=True)
        else:
            self._print_result(runner_outputs)

    @staticmethod
    def _print_result(runner_outputs: object) -> None:
        """Print workflow result. Mirrors ConsoleFrontEndPlugin output formatting."""
        line = "-" * 50
        prefix = f"{line}\n{Fore.GREEN}\u2705 Workflow Result:\n"
        suffix = f"{Style.RESET_ALL}\n{line}"

        logger.info("%s%s%s", prefix, runner_outputs, suffix)

        has_info_stream_handler = any(
            isinstance(h, logging.StreamHandler) and h.level <= logging.INFO
            for h in logging.getLogger().handlers
        )
        if not has_info_stream_handler:
            print(f"{prefix}{runner_outputs}{suffix}")
