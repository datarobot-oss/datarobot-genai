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

"""
DataRobot Tools Library.

A reusable library for building tools with DataRobot integration.

All subpackages are importable from this module, for example::

    from datarobot_genai.drtools import confluence
    from datarobot_genai.drtools import dr_docs
    from datarobot_genai.drtools import predictive
"""

from datarobot_genai.drtools import clients
from datarobot_genai.drtools import confluence
from datarobot_genai.drtools import dr_docs
from datarobot_genai.drtools import gdrive
from datarobot_genai.drtools import jira
from datarobot_genai.drtools import microsoft_graph
from datarobot_genai.drtools import perplexity
from datarobot_genai.drtools import predictive
from datarobot_genai.drtools import tavily

__all__ = [
    "clients",
    "confluence",
    "dr_docs",
    "gdrive",
    "jira",
    "microsoft_graph",
    "perplexity",
    "predictive",
    "tavily",
]
