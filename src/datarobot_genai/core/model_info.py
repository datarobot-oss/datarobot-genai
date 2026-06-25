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

"""``litellm.get_model_info`` for DataRobot-routed models.

litellm has no ``datarobot/*`` entries, so this strips the ``datarobot/`` prefix
and resolves the inner provider; other models pass through. Read capabilities
(``supports_function_calling``, ``supports_reasoning``, …) off the result as you
would from ``litellm.get_model_info``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo


def get_model_info(model: str) -> ModelInfo:
    """``litellm.get_model_info``, resolving the ``datarobot/`` prefix.

    Azure deployment names dot their dashed version to litellm's key
    (``gpt-5-1`` -> ``gpt-5.1``).
    """
    import litellm  # noqa: PLC0415 — lazy so importing this module stays cheap

    if model.startswith("datarobot/"):
        model = model.removeprefix("datarobot/")
        if model.startswith("azure/"):
            model = re.sub(r"(gpt-\d)-(\d)(?=-|$)", r"\1.\2", model)  # gpt-5-1 -> gpt-5.1
    return litellm.get_model_info(model)
