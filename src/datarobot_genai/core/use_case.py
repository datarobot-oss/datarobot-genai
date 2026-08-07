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

"""Resolve a DataRobot use case by name, creating one if needed."""

from __future__ import annotations

import logging

import datarobot as dr

logger = logging.getLogger(__name__)


def find_or_create_use_case(name: str) -> str | None:
    """Return the id of the use case named ``name``, creating one if absent.

    Creating writes to the caller's DataRobot org. Never raises; returns
    ``None`` on any SDK error.
    """
    try:
        # `search` is a substring match, so require an exact name.
        matches = [uc for uc in dr.UseCase.list(search_params={"search": name}) if uc.name == name]
        use_case = (
            matches[0]
            if matches
            else dr.UseCase.create(
                name=name,
                description=f"Created by datarobot-genai find_or_create_use_case({name!r}).",
            )
        )
        return str(use_case.id)
    except Exception:
        # Never raise: this is optional convenience.
        logger.exception("Failed to find or create use case %r", name)
        return None
