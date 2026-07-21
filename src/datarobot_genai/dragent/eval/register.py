# Copyright 2026 DataRobot, Inc. and its affiliates.
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
"""NAT plugin entry: registers all ``dr_*`` evaluators on import."""

from datarobot_genai.dragent.eval import agent_goal_accuracy as _agent_goal_accuracy  # noqa: F401
from datarobot_genai.dragent.eval import faithfulness as _faithfulness  # noqa: F401
from datarobot_genai.dragent.eval import guideline_adherence as _guideline_adherence  # noqa: F401
from datarobot_genai.dragent.eval import task_adherence as _task_adherence  # noqa: F401

__all__ = [
    "_agent_goal_accuracy",
    "_faithfulness",
    "_guideline_adherence",
    "_task_adherence",
]
