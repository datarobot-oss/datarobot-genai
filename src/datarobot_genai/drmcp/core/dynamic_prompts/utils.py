# Copyright 2025 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from uuid import uuid4

from fastmcp import FastMCP
from fastmcp.exceptions import NotFoundError

_SUFFIX_LENGTH: int = 4


async def get_prompt_name_no_duplicate(mcp: FastMCP, prompt_name: str) -> str:
    """Handle prompt name duplicate.

    We're working optimistic here -- we're keeping default names unless there's collision
    """
    try:
        prompt = await mcp.get_prompt(prompt_name)
    except NotFoundError:
        return prompt_name

    prompt_name_suffix = str(uuid4())[:_SUFFIX_LENGTH]
    return f"{prompt.name} ({prompt_name_suffix})"
