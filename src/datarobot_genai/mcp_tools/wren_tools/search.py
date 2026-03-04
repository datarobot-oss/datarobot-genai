"""Web search tool using Perplexity AI.

Ported from wren-mcp web_search.py. Only enabled when a Perplexity API
key is present in the environment.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from datarobot_genai.mcp_tools._registry import register_tool

logger = logging.getLogger(__name__)


async def web_search(
    query: str,
    recency_filter: str | None = None,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search the web using Perplexity AI.

    recency_filter options: 'day', 'week', 'month', 'year', or None for all time.
    Requires PERPLEXITY_API_KEY environment variable to be set.
    Returns answer, citations, and search_results.
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        return {
            "error": "PERPLEXITY_API_KEY not configured. Set this environment variable to enable web search.",
            "answer": None,
            "citations": [],
            "search_results": [],
        }

    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": "sonar",
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 1024,
        "return_citations": True,
        "return_related_questions": False,
    }
    if recency_filter:
        payload["search_recency_filter"] = recency_filter

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.perplexity.ai/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    answer = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    citations = data.get("citations", [])
    search_results = [
        {"url": c, "index": i + 1} for i, c in enumerate(citations[:max_results])
    ]

    return {
        "answer": answer,
        "citations": citations,
        "search_results": search_results,
    }


if os.environ.get("PERPLEXITY_API_KEY"):
    register_tool(
        "web_search",
        web_search,
        "Search the web using Perplexity AI. Requires PERPLEXITY_API_KEY.",
        "wren_tools",
    )
else:
    # Register anyway so the tool exists; it will return a helpful error at runtime
    register_tool(
        "web_search",
        web_search,
        "Search the web using Perplexity AI. Requires PERPLEXITY_API_KEY env var.",
        "wren_tools",
    )
