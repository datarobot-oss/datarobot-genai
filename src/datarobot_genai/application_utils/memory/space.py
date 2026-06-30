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

"""``DRMemorySpace`` — the entry point for all Memory Service ORM operations.

A memory space scopes every session and event call.  Create or fetch one,
then pass it to ``DRSession.post`` / ``DRSession.get``.

Examples
--------
.. code-block:: python

    import asyncio
    from datarobot_genai.application_utils.memory import (
        DRMemorySpace,
        MemoryServiceClient,
    )

    async def main() -> None:
        async with MemoryServiceClient() as client:
            # Create a new space (or adopt the existing one with the same key)
            space = await DRMemorySpace.post(
                client,
                description="My agent memory",
                deduplication_key="my-agent-space-v1",
            )
            print(space.id)

            # Fetch an existing space by id
            space2 = await DRMemorySpace.get(client, space.id)
            assert space2.id == space.id

    asyncio.run(main())
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from datarobot_genai.application_utils.memory.exceptions import MemoryConflictError

if TYPE_CHECKING:
    from datarobot_genai.application_utils.memory._client import MemoryServiceClient


class DRMemorySpace:
    """Represents a DataRobot Agentic Memory Service memory space.

    Acts as the container for sessions; every session and event call is scoped
    to a space.  Construct via the class methods :meth:`post` or :meth:`get`
    rather than directly instantiating.

    Attributes
    ----------
    description : str | None
        Human-readable description of the space.
    deduplication_key : str | None
        Unique client-assigned key; enables idempotent space creation.
    llm_model_name : str | None
        LLM model name for ``extract_memories`` lifecycle strategies.
    llm_base_url : str | None
        LLM base URL override.
    custom_instructions : str | None
        Custom instructions passed to the LLM when extracting memories.
    created_at : str
        ISO-8601 creation timestamp (server-assigned).
    """

    def __init__(
        self,
        client: MemoryServiceClient,
        *,
        id: str,
        user_id: str,
        tenant_id: str,
        description: str | None = None,
        deduplication_key: str | None = None,
        llm_model_name: str | None = None,
        llm_base_url: str | None = None,
        custom_instructions: str | None = None,
        created_at: str = "",
    ) -> None:
        self._client = client
        self._id = id
        self._user_id = user_id
        self._tenant_id = tenant_id
        self.description = description
        self.deduplication_key = deduplication_key
        self.llm_model_name = llm_model_name
        self.llm_base_url = llm_base_url
        self.custom_instructions = custom_instructions
        self.created_at = created_at

    # ── Read-only server-assigned properties ──────────────────────────────

    @property
    def id(self) -> str:
        """Server-assigned memory space UUID."""
        return self._id

    @property
    def user_id(self) -> str:
        """Owner user ID."""
        return self._user_id

    @property
    def tenant_id(self) -> str:
        """Tenant UUID."""
        return self._tenant_id

    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        return (
            f"DRMemorySpace(id={self._id!r}, description={self.description!r}, "
            f"deduplication_key={self.deduplication_key!r})"
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    @classmethod
    def _from_wire(cls, client: MemoryServiceClient, data: dict[str, Any]) -> DRMemorySpace:
        """Construct a ``DRMemorySpace`` from a wire response dict."""
        return cls(
            client,
            id=data["memorySpaceId"],
            user_id=data.get("userId", ""),
            tenant_id=data.get("tenantId", ""),
            description=data.get("description"),
            deduplication_key=data.get("deduplicationKey"),
            llm_model_name=data.get("llmModelName"),
            llm_base_url=data.get("llmBaseUrl"),
            custom_instructions=data.get("customInstructions"),
            created_at=str(data.get("createdAt", "")),
        )

    def _update_from_wire(self, data: dict[str, Any]) -> None:
        """Update mutable attributes in-place from a wire response dict."""
        self.description = data.get("description")
        self.deduplication_key = data.get("deduplicationKey")
        self.llm_model_name = data.get("llmModelName")
        self.llm_base_url = data.get("llmBaseUrl")
        self.custom_instructions = data.get("customInstructions")
        self.created_at = str(data.get("createdAt", self.created_at))

    # ── Class-method operations ───────────────────────────────────────────

    @classmethod
    async def post(
        cls,
        client: MemoryServiceClient,
        *,
        description: str | None = None,
        deduplication_key: str | None = None,
        llm_model_name: str | None = None,
        llm_base_url: str | None = None,
        custom_instructions: str | None = None,
    ) -> DRMemorySpace:
        """Create a new memory space, or adopt an existing one on a dedupliction conflict.

        If a ``deduplication_key`` is supplied and a space with that key already
        exists, the existing space is fetched and returned (409 → adopt).

        Parameters
        ----------
        client : MemoryServiceClient
            Transport client.
        description : str | None
            Human-readable description (max 1000 chars).
        deduplication_key : str | None
            Unique client key for idempotent creation (1–72 chars).
        llm_model_name : str | None
            LLM model name for ``extract_memories`` strategies.
        llm_base_url : str | None
            LLM base URL override.
        custom_instructions : str | None
            Custom LLM instructions (max 10 000 chars).

        Returns
        -------
        DRMemorySpace
            The newly created or adopted memory space.
        """
        payload: dict[str, Any] = {}
        if description is not None:
            payload["description"] = description
        if deduplication_key is not None:
            payload["deduplicationKey"] = deduplication_key
        if llm_model_name is not None:
            payload["llmModelName"] = llm_model_name
        if llm_base_url is not None:
            payload["llmBaseUrl"] = llm_base_url
        if custom_instructions is not None:
            payload["customInstructions"] = custom_instructions

        try:
            resp = await client.request("POST", "new/", json=payload)
            return cls._from_wire(client, resp.json())
        except MemoryConflictError as exc:
            if exc.existing_id:
                resp = await client.request("GET", f"{exc.existing_id}/")
                return cls._from_wire(client, resp.json())
            raise

    @classmethod
    async def get(cls, client: MemoryServiceClient, space_id: str) -> DRMemorySpace:
        """Fetch a memory space by its server-assigned ID.

        Parameters
        ----------
        client : MemoryServiceClient
            Transport client.
        space_id : str
            UUID of the memory space.

        Returns
        -------
        DRMemorySpace

        Raises
        ------
        MemoryNotFoundError
            If no space with the given ID exists (or it belongs to another user).
        """
        resp = await client.request("GET", f"{space_id}/")
        return cls._from_wire(client, resp.json())

    @classmethod
    async def list(
        cls,
        client: MemoryServiceClient,
        *,
        deduplication_key: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[DRMemorySpace]:
        """List memory spaces visible to the authenticated user.

        Parameters
        ----------
        client : MemoryServiceClient
            Transport client.
        deduplication_key : str | None
            Exact-match filter on ``deduplicationKey``.
        offset : int
            Number of spaces to skip (for pagination).
        limit : int
            Maximum number of spaces to return (1–100).

        Returns
        -------
        list[DRMemorySpace]
        """
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if deduplication_key is not None:
            params["deduplicationKey"] = deduplication_key
        resp = await client.request("GET", "", params=params)
        data = resp.json()
        return [cls._from_wire(client, item) for item in data.get("items", [])]

    # ── Instance operations ───────────────────────────────────────────────

    async def patch(
        self,
        *,
        description: str | None = None,
        llm_model_name: str | None = None,
        llm_base_url: str | None = None,
        custom_instructions: str | None = None,
    ) -> None:
        """Update this memory space in place.

        Only the supplied keyword arguments are changed; omitted fields keep
        their current values on the server.

        Parameters
        ----------
        description : str | None
            New description.
        llm_model_name : str | None
            New LLM model name.
        llm_base_url : str | None
            New LLM base URL.
        custom_instructions : str | None
            New custom instructions.
        """
        payload: dict[str, Any] = {}
        if description is not None:
            payload["description"] = description
        if llm_model_name is not None:
            payload["llmModelName"] = llm_model_name
        if llm_base_url is not None:
            payload["llmBaseUrl"] = llm_base_url
        if custom_instructions is not None:
            payload["customInstructions"] = custom_instructions

        resp = await self._client.request("PATCH", f"{self._id}/", json=payload)
        self._update_from_wire(resp.json())

    async def delete(self) -> None:
        """Soft-delete this memory space.

        After deletion the space is no longer accessible via ``get`` or ``list``.
        """
        await self._client.request("DELETE", f"{self._id}/")
