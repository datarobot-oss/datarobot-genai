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

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Literal


class MetadataBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool, for the LLM to identify and call the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """The description of the tool, for the LLM to understand its purpose
        and all additional instructions and context about the tool, which can
        help the LLM to better utilize the tool in the right context.
        """
        pass

    @property
    @abstractmethod
    def method(self) -> Literal["GET", "POST", "PATCH", "PUT", "DELETE"]:
        """HTTP method to use when calling the tool, e.g. `POST` or `GET` etc."""
        pass

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """The endpoint path of the tool, e.g. `/weather/{city}/forecast`."""
        pass

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """The JSON schema defining the input parameters for the tool.

        Structure:
            {
                "type": "object",
                "properties": {
                    "path_params": {...},    # Optional: path parameter schemas
                    "query_params": {...},   # Optional: query parameter schemas
                    "data": {...},           # Optional: form/body data schema
                    "json": {...}            # Optional: JSON body schema
                },
                "required": [...]            # Optional: required properties
            }
        """
        pass

    @property
    @abstractmethod
    def headers(self) -> dict[str, str]:
        """Optional HTTP headers to include when calling the tool."""
        pass
