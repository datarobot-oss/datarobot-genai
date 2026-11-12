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

"""OTLP HTTP export authenticated with the workload identity token.

genai's own exporters take ``workload_otlp_session()`` as ``session=``. Exporters
built elsewhere (NAT's) load ``otlp_session`` through the OTel SDK's
credential-provider hook, ``OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER``.
"""

from __future__ import annotations

import logging
import os
from importlib.metadata import entry_points

import requests

from datarobot_genai.core.auth.workload_token import WorkloadTokenAuth
from datarobot_genai.core.auth.workload_token import WorkloadTokenError
from datarobot_genai.core.auth.workload_token import get_workload_token_source

logger = logging.getLogger(__name__)

OTLP_CREDENTIAL_PROVIDER_ENV = "OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER"
OTLP_CREDENTIAL_PROVIDER_GROUP = "opentelemetry_otlp_credential_provider"
OTLP_CREDENTIAL_PROVIDER_NAME = "datarobot_workload_identity"


class _BestEffortWorkloadTokenAuth(WorkloadTokenAuth):
    """Sends without the bearer when no token is held; the source already logged why.

    Static ``X-DataRobot-Api-Key`` headers still authenticate. Remove with them.
    """

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        try:
            return super().__call__(r)
        except WorkloadTokenError:
            return r


def workload_otlp_session() -> requests.Session | None:
    """Build a session that adds the workload bearer token; ``None`` without identity."""
    source = get_workload_token_source()
    if source is None:
        return None
    session = requests.Session()
    session.auth = _BestEffortWorkloadTokenAuth(source)
    return session


def otlp_session() -> requests.Session:
    """``opentelemetry_otlp_credential_provider`` entry point; must return a Session."""
    return workload_otlp_session() or requests.Session()


def enable_otlp_credential_provider() -> bool:
    """Point OTLP HTTP exporters built without ``session=`` at ``otlp_session``.

    Covers exporters genai does not construct itself (e.g. NAT's). Leaves an
    existing setting alone and is a no-op without workload identity, or when the
    entry point is not installed (the SDK would raise on an unknown name).
    """
    if os.environ.get(OTLP_CREDENTIAL_PROVIDER_ENV):
        return False
    if get_workload_token_source() is None:
        return False
    if not entry_points(group=OTLP_CREDENTIAL_PROVIDER_GROUP, name=OTLP_CREDENTIAL_PROVIDER_NAME):
        logger.debug("OTLP credential provider entry point not installed")
        return False
    os.environ[OTLP_CREDENTIAL_PROVIDER_ENV] = OTLP_CREDENTIAL_PROVIDER_NAME
    return True
