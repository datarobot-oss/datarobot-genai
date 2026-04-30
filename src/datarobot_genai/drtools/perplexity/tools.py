# Copyright 2026 DataRobot, Inc.
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

"""Perplexity MCP tools."""

import logging
from typing import Annotated
from typing import Any
from typing import Literal

from datarobot_genai.drtools.core import tool_metadata
from datarobot_genai.drtools.core.clients.perplexity import MAX_QUERIES
from datarobot_genai.drtools.core.clients.perplexity import MAX_RESULTS
from datarobot_genai.drtools.core.clients.perplexity import MAX_RESULTS_DEFAULT
from datarobot_genai.drtools.core.clients.perplexity import MAX_SEARCH_DOMAIN_FILTER
from datarobot_genai.drtools.core.clients.perplexity import MAX_TOKENS_PER_PAGE
from datarobot_genai.drtools.core.clients.perplexity import MAX_TOKENS_PER_PAGE_DEFAULT
from datarobot_genai.drtools.core.clients.perplexity import PerplexityClient
from datarobot_genai.drtools.core.clients.perplexity import get_perplexity_access_token
from datarobot_genai.drtools.core.exceptions import ToolError
from datarobot_genai.drtools.core.exceptions import ToolErrorKind

logger = logging.getLogger(__name__)

_PPLX_SEARCH_GUIDE = "https://docs.perplexity.ai/guides/search"
_PPLX_STRUCTURED = "https://docs.perplexity.ai/guides/structured-outputs"


@tool_metadata(
    tags={"perplexity", "web", "search", "websearch", "daria"},
    description=(
        "[Perplexity—web search] Use when the user needs ranked web sources/snippets for a "
        "question or multi-part research (string or list of sub-queries), with optional domain "
        "allow/deny and recency. Not long-form single narrative answers (perplexity_think), not "
        "DataRobot docs (search_datarobot_agentic_docs), not raw URL extraction "
        "(tavily_extract).\n\n"
        f"query may be a string or up to {MAX_QUERIES} strings; tune max_results (1-{MAX_RESULTS}) "
        f"and max_tokens_per_page (1-{MAX_TOKENS_PER_PAGE}).\n\n"
        f"Reference: {_PPLX_SEARCH_GUIDE}"
    ),
)
async def perplexity_search(
    *,
    query: Annotated[
        str,
        list[str],
        f"The search query string OR "
        f"a list of up to {MAX_QUERIES} sub-queries for multi-query research.",
    ],
    search_domain_filter: Annotated[
        list[str] | None,
        f"Up to {MAX_SEARCH_DOMAIN_FILTER} domains/URLs "
        f"to allowlist or denylist (prefix with '-').",
    ] = None,
    recency: Annotated[
        Literal["day", "week", "month", "year"] | None,
        "Optional. Pass exactly one of these strings or omit: 'day', 'week', 'month', 'year' "
        "to filter by recency.",
    ] = None,
    max_results: Annotated[
        int, f"Number of ranked results to return (1-{MAX_RESULTS})."
    ] = MAX_RESULTS_DEFAULT,
    max_tokens_per_page: Annotated[
        int,
        f"Content extraction cap per page (1-{MAX_TOKENS_PER_PAGE}) "
        f"(default {MAX_TOKENS_PER_PAGE_DEFAULT}).",
    ] = MAX_TOKENS_PER_PAGE_DEFAULT,
) -> dict[str, Any]:
    # Check if query is empty or None
    if not query:
        raise ToolError(
            "Argument validation error: query cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    # Handle string queries
    if isinstance(query, str):
        if not query.strip():
            raise ToolError(
                "Argument validation error: query cannot be empty or whitespace only.",
                kind=ToolErrorKind.VALIDATION,
            )
    # Handle list queries
    elif isinstance(query, list):
        if not query:  # Empty list
            raise ToolError(
                "Argument validation error: query list cannot be empty.",
                kind=ToolErrorKind.VALIDATION,
            )
        if len(query) > MAX_QUERIES:
            raise ToolError(
                f"Argument validation error: query list cannot be bigger than {MAX_QUERIES}.",
                kind=ToolErrorKind.VALIDATION,
            )
        if not all(q.strip() for q in query):
            raise ToolError(
                "Argument validation error: query cannot contain empty strings.",
                kind=ToolErrorKind.VALIDATION,
            )
    if search_domain_filter and len(search_domain_filter) > MAX_SEARCH_DOMAIN_FILTER:
        raise ToolError(
            f"Argument validation error: "
            f"maximum number of search domain filters is {MAX_SEARCH_DOMAIN_FILTER}.",
            kind=ToolErrorKind.VALIDATION,
        )
    if max_results <= 0:
        raise ToolError(
            "Argument validation error: max_results must be greater than 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    if max_results > MAX_RESULTS:
        raise ToolError(
            f"Argument validation error: "
            f"max_results must be smaller than or equal to {MAX_RESULTS}.",
            kind=ToolErrorKind.VALIDATION,
        )
    if max_tokens_per_page <= 0:
        raise ToolError(
            "Argument validation error: max_tokens_per_page must be greater than 0.",
            kind=ToolErrorKind.VALIDATION,
        )
    if max_tokens_per_page > MAX_TOKENS_PER_PAGE:
        raise ToolError(
            f"Argument validation error: "
            f"max_tokens_per_page must be smaller than or equal to {MAX_TOKENS_PER_PAGE}.",
            kind=ToolErrorKind.VALIDATION,
        )

    access_token = await get_perplexity_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with PerplexityClient(access_token=access_token) as perplexity_client:
        results = await perplexity_client.search(
            query=query,
            search_domain_filter=search_domain_filter,
            recency=recency,
            max_results=max_results,
            max_tokens_per_page=max_tokens_per_page,
        )

    return {
        "results": results,
        "count": len(results),
        "metadata": {
            "queriesExecuted": len(query) if isinstance(query, list) else 1,
            "filtersApplied": {"domains": search_domain_filter, "recency": recency},
            "extractionLimit": max_tokens_per_page,
        },
    }


@tool_metadata(
    tags={"perplexity", "think", "research", "answer", "daria"},
    description=(
        "[Perplexity—answer / reason] Use when the user wants a model-authored reply with "
        "citations: fast Q&A, deeper reasoning, or long research report (set model to one of "
        "the string literals on the parameter), "
        "optionally JSON-shaped output via schema. Not primarily for ranked URL lists "
        "(perplexity_search), not internal DR docs.\n\n"
        "Models: sonar (fast), sonar-reasoning-pro (complex logic), sonar-deep-research (reports). "
        "Pass json_schema for structured extraction.\n\n"
        f"Reference: {_PPLX_STRUCTURED}"
    ),
)
async def perplexity_think(
    *,
    prompt: Annotated[str, "The research prompt or instruction."],
    model: Annotated[
        Literal["sonar", "sonar-reasoning-pro", "sonar-deep-research"],
        "Pass exactly one of these strings (hyphens as shown): 'sonar', "
        "'sonar-reasoning-pro', 'sonar-deep-research'. Fast Q&A vs deeper reasoning vs long "
        "research; omit to default to 'sonar'.",
    ] = "sonar",
    json_schema: Annotated[
        dict | None, "Optional JSON Schema to enforce structured output for data extraction."
    ] = None,
) -> dict[str, Any]:
    if not prompt or not prompt.strip():
        raise ToolError(
            "Argument validation error: prompt cannot be empty.", kind=ToolErrorKind.VALIDATION
        )

    access_token = await get_perplexity_access_token()
    if isinstance(access_token, ToolError):
        raise access_token

    async with PerplexityClient(access_token=access_token) as perplexity_client:
        result = await perplexity_client.think(prompt=prompt, model=model, json_schema=json_schema)

    return {
        "model": model,
        "citations": result.citations,
        "usage": result.usage.as_flat_dict() if result.usage else None,
        "content": result.answer,
    }
