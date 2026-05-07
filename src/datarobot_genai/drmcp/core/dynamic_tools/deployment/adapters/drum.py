# Copyright 2026 DataRobot, Inc.
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

"""Backwards-compatible shim — implementation moved to ``drtools.dynamic.adapters.drum``."""

from datarobot_genai.drtools.dynamic.adapters.drum import DrumMetadataAdapter
from datarobot_genai.drtools.dynamic.adapters.drum import DrumTargetType
from datarobot_genai.drtools.dynamic.adapters.drum import get_default_schema
from datarobot_genai.drtools.dynamic.adapters.drum import is_drum

__all__ = [
    "DrumMetadataAdapter",
    "DrumTargetType",
    "get_default_schema",
    "is_drum",
]
