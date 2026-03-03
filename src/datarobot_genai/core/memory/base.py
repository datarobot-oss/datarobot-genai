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

from abc import ABC
from abc import abstractmethod


class BaseMemoryClient(ABC):
    @abstractmethod
    def retrieve(
        self,
        user_id: str,
        prompt: str,
    ) -> str:
        """Return relevant memory context as plain text."""
        raise NotImplementedError

    @abstractmethod
    def store(
        self,
        user_id: str,
        user_message: str,
    ) -> None:
        """Persist conversation interaction."""
        raise NotImplementedError
