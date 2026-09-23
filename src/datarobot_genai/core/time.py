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
from enum import Enum, auto


class TimeMeasurement(Enum):
    HOUR = auto()
    MINUTE = auto()
    SECOND = auto()

    def to_numeric_value_in_second(self) -> int:
        return {
            TimeMeasurement.HOUR: 3600,
            TimeMeasurement.MINUTE: 60,
            TimeMeasurement.SECOND: 1,
        }[self]
