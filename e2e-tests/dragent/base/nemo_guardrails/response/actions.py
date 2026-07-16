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
import os

from langchain_core.language_models.llms import BaseLLM
from nemoguardrails.actions import action
from nemoguardrails.actions.llm.utils import llm_call
from nemoguardrails.context import llm_call_info_var
from nemoguardrails.llm.taskmanager import LLMTaskManager
from nemoguardrails.llm.types import Task
from nemoguardrails.logging.explain import LLMCallInfo

BLOCKED_TERMS_FILE = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../blocked_terms.txt")
)
log = logging.getLogger(__name__)


@action(is_system_action=True)
async def check_blocked_terms(context: dict | None = None):
    bot_message = context.get("bot_message")

    with open(BLOCKED_TERMS_FILE) as f:
        # Read the contents of the blocked terms from the blocked term txt file
        blocked_terms = f.read().lower().strip().split("\n")

    for term in blocked_terms:
        if term in bot_message.lower():
            return True

    return False


@action(is_system_action=True)
async def self_check_output(
    llm_task_manager: LLMTaskManager,
    context: dict | None = None,
    llm: BaseLLM | None = None,
):
    """Checks if the output from the bot.

    Prompt the LLM, using the `self_check_output` task prompt, to determine if the output
    from the bot should be allowed or not.

    The LLM call should return "yes" if the output is bad and should be blocked
    (this is consistent with self_check_input_prompt).

    Returns
    -------
        True if the output should be allowed, False otherwise.
    """  # noqa: D401
    bot_response = context.get("bot_message")
    user_input = context.get("user_message")
    if bot_response:
        prompt = llm_task_manager.render_task_prompt(
            task=Task.SELF_CHECK_OUTPUT,
            context={
                "user_input": user_input,
                "bot_response": bot_response,
            },
        )

        # Initialize the LLMCallInfo object
        llm_call_info_var.set(LLMCallInfo(task=Task.SELF_CHECK_OUTPUT.value))

        response = await llm_call(llm, prompt, llm_params={"temperature": 0.0})

        # nemoguardrails 0.23: llm_call returns LLMResponse, not str.
        response = getattr(response, "content", response).lower().strip()
        log.info(f"Output self-checking result is: `{response}`.")

        if "yes" in response:
            return False

    return True
