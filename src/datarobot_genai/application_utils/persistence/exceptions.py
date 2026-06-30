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

"""Typed exceptions raised by the Memory Service ORM.

All exceptions derive from ``DRMemoryServiceError``.  The naming deliberately
avoids ``MemoryError`` to prevent shadowing the Python built-in.

Exception hierarchy
-------------------
::

    DRMemoryServiceError
    ├── DRMemoryNotFoundError        (HTTP 404)
    ├── DRMemoryBadRequestError      (HTTP 400)
    ├── DRMemoryValidationError      (HTTP 422, schema validation)
    ├── DRMemoryConflictError        (HTTP 409, deduplication conflict)
    └── DRMemoryVersionConflictError (HTTP 409 session stale / 422 event stale)
"""

from __future__ import annotations

from typing import Any


class DRMemoryServiceError(Exception):
    """Base exception for all Memory Service ORM errors.

    Parameters
    ----------
    detail : str
        Human-readable error detail.
    status_code : int | None
        HTTP status code, if applicable.
    payload : dict | None
        Raw response body, if available.
    """

    def __init__(
        self,
        detail: str,
        *,
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code
        self.payload = payload

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return f"{type(self).__name__}({self.detail!r}, status_code={self.status_code})"


class DRMemoryNotFoundError(DRMemoryServiceError):
    """Raised when the requested resource does not exist (HTTP 404)."""


class DRMemoryBadRequestError(DRMemoryServiceError):
    """Raised on client-side validation errors from the service (HTTP 400).

    Common causes: an event emitter is not a session participant, an invalid
    ObjectId is supplied for a participant filter.
    """


class DRMemoryValidationError(DRMemoryServiceError):
    """Raised on schema validation errors from the service (HTTP 422).

    This typically means the request body or query parameters failed the
    service's Pydantic validation (e.g. a field is too long, a required field
    is missing, or mutually exclusive parameters are both supplied).
    """


class DRMemoryConflictError(DRMemoryServiceError):
    """Raised on a deduplication conflict when creating a session or ``DRMemorySpace`` (HTTP 409).

    The ORM automatically adopts the existing resource (by fetching it via
    ``existing_id``) rather than propagating this exception to callers in
    the normal ``post()`` flow.

    Parameters
    ----------
    existing_id : str | None
        Server-assigned ID of the existing resource.
    location : str | None
        URL from the service ``Location`` header pointing to the existing resource.
    """

    def __init__(
        self,
        detail: str,
        *,
        status_code: int | None = None,
        payload: dict[str, Any] | None = None,
        existing_id: str | None = None,
        location: str | None = None,
    ) -> None:
        super().__init__(detail, status_code=status_code, payload=payload)
        self.existing_id = existing_id
        self.location = location


class DRMemoryVersionConflictError(DRMemoryServiceError):
    """Raised on an optimistic-concurrency failure.

    Surfaces as HTTP 409 for session ``patch()`` (stale ``If-Match`` header)
    and as HTTP 422 for event ``patch()`` (stale ``createdAt`` token).

    Resolution: re-read the resource to get the current version, then retry.
    """
