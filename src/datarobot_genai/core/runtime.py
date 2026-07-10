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

"""DataRobot hosted-runtime detection helpers.

Canonical helpers for determining whether the current process is running
inside a DataRobot-managed container (workload or custom-model deployment)
and retrieving the platform-injected identifiers.
"""

from __future__ import annotations

import os

WORKLOAD_ID_ENV = "WORKLOAD_ID"
DEPLOYMENT_ID_ENV = "MLOPS_DEPLOYMENT_ID"


def get_workload_id() -> str | None:
    """Return the platform-injected workload ID, or None when not on a workload."""
    return os.environ.get(WORKLOAD_ID_ENV, "").strip() or None


def get_deployment_id() -> str | None:
    """Return the platform-injected deployment ID, or None when not on a deployment."""
    return os.environ.get(DEPLOYMENT_ID_ENV, "").strip() or None


def is_workload_mode() -> bool:
    """Return True when running inside a DataRobot workload container."""
    return get_workload_id() is not None


def is_deployment_mode() -> bool:
    """Return True when running inside a DataRobot custom-model deployment container."""
    return get_deployment_id() is not None


def is_hosted_runtime() -> bool:
    """Return True when running in any DataRobot-hosted runtime (vs local dev)."""
    return is_workload_mode() or is_deployment_mode()
