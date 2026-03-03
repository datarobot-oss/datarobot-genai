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

from mem0 import MemoryClient

from .base import BaseMemoryClient


class Mem0Client(BaseMemoryClient):
    def __init__(self, api_key: str):
        self._memory = MemoryClient(api_key=api_key)

    def retrieve(self, user_id: str, prompt: str) -> str:
        filters = {"OR": [{"user_id": user_id}]}

        user_memories = self._memory.search(
            query=prompt, filters=filters, version="v2", output_format="v1.1"
        )
        return user_memories

    def store(
        self,
        user_id: str,
        user_message: str,
    ) -> None:
        messages = [{"role": "user", "content": user_message}]

        self._memory.add(messages, user_id=user_id, version="v2", output_format="v1.1")
