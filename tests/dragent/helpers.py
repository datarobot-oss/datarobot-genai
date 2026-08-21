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

"""Helpers shared across the dragent test packages."""

from typing import Any

import jwt

# Nothing under test verifies signatures (the API Gateway owns that), so the key is
# arbitrary; it is sized past 32 bytes only to keep PyJWT from warning about a short HMAC key.
_SIGNING_KEY = "test-signing-key-long-enough-for-hs256"


def make_jwt(**claims: Any) -> str:
    """Build a signed JWT carrying ``claims``."""
    return jwt.encode(claims, _SIGNING_KEY, algorithm="HS256")
