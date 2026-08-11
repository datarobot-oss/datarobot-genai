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

from unittest.mock import MagicMock

from nat.data_models.user_info import UserInfo

AUTH_HANDLER_PATH = "datarobot_genai.dragent.frontends.session._auth_handler.get_context"


def expected_workflow_key(raw_user_id: str) -> str:
    """Compute the expected UUID5 workflow key for a raw DataRobot user ID."""
    return UserInfo._from_session_cookie(raw_user_id).get_user_id()


def make_auth_ctx(user_id: str) -> MagicMock:
    """Build a mock AuthCtx with the given ``user.id``."""
    ctx = MagicMock()
    ctx.user.id = user_id
    return ctx
