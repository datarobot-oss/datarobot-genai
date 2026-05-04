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

from typing import Annotated

# This is a dummy tool we use to test that agents does not hallucinate tool calls. It only accepts
# one value for the type argument, and it only returns a fixed value we can check in a test


def generate_objectid(
    type: Annotated[
        str,
        """The type of object to generate an ID for. Should be only 'deployment',
        all other values will cause an error""",
    ],
) -> str:
    """Generate a unique object ID for a deployment."""
    if type != "deployment":
        raise ValueError("Invalid type. Should be only 'deployment'")

    return "69cbb73789723b6936c6c9e1"
