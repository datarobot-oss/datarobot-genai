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

import contextvars
from collections.abc import Callable
from collections.abc import Coroutine
from typing import Annotated
from typing import Any
from typing import TypeVar

from .strategy import HumanInTheLoopStorageStrategy

T = TypeVar("T")

# This value has to be set by the framework adapter
current_tool_call_id_var = contextvars.ContextVar[str]


def get_human_in_the_loop_tool(
    strategy: HumanInTheLoopStorageStrategy[T],
) -> Callable[[str], Coroutine[Any, Any, T]]:
    async def request_human_input(
        prompt: Annotated[str, "The prompt to request human input for"],
    ) -> T:
        """Request human input for the given object."""
        _ = prompt  # Not necessary to use here, AG-UI will pass it to the client
        try:
            id = current_tool_call_id_var.get()  # type: ignore[call-overload]
        except LookupError:
            raise ValueError("No tool call ID set. Must be set by the framework adapter.")

        return await strategy.wait_for_human_input(id)

    return request_human_input
