# Copyright 2025 DataRobot, Inc.
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

"""DataRobot Agentic AI Documentation search client.

Searches the DataRobot agentic-ai documentation section without requiring
any API keys. Parses the sitemap.xml to discover agentic-ai documentation
page URLs, fetches their full HTML content concurrently at index-build time,
and builds a TF-IDF index from real page titles and body text. The index is
cached in memory for fast repeated queries.

Only pages under https://docs.datarobot.com/en/docs/agentic-ai/ are indexed,
keeping the corpus small (~28 pages) so the full index can be built in a few
seconds at server startup.
"""

import asyncio
import logging
import math
import re
import time
from collections import Counter
from html.parser import HTMLParser
from typing import Any
from xml.etree import ElementTree

import aiohttp

logger = logging.getLogger(__name__)

DOCS_BASE_URL = "https://docs.datarobot.com"
DOCS_SITEMAP_URL = f"{DOCS_BASE_URL}/en/docs/sitemap.xml"

# Only index pages under this path — keeps the corpus small enough to fetch
# all content at build time without a background enrichment job.
AGENTIC_AI_PATH = "/en/docs/agentic-ai/"

# Cache TTL in seconds (1 day)
CACHE_TTL_SECONDS = 86400

# Maximum number of results to return
MAX_RESULTS = 20
MAX_RESULTS_DEFAULT = 5

# Request timeout in seconds
REQUEST_TIMEOUT_SECONDS = 30

# Maximum number of concurrent HTTP requests to fetch content
CONTENT_FETCH_CONCURRENCY = 30


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, stripping non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _compute_tf(tokens: list[str]) -> dict[str, float]:
    """Compute term frequency for a list of tokens."""
    counts = Counter(tokens)
    total = len(tokens) if tokens else 1
    return {term: count / total for term, count in counts.items()}


class _ContentExtractor(HTMLParser):
    """Extract meaningful text content from an HTML page."""

    # Tags whose text content we want to capture
    _CONTENT_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "dt", "dd"}
    # Tags to skip entirely
    _SKIP_TAGS = {"script", "style", "nav", "footer", "header"}

    def __init__(self) -> None:
        super().__init__()
        self._texts: list[str] = []
        self._title: str = ""
        self._in_title = False
        self._skip_depth = 0
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._tag_stack.append(tag)
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self._title = text
        if self._skip_depth == 0 and text:
            self._texts.append(text)

    @property
    def title(self) -> str:
        return self._title

    @property
    def text_content(self) -> str:
        return " ".join(self._texts)


class DocPage:
    """Represents a single documentation page."""

    def __init__(self, url: str, title: str, text: str = "") -> None:
        self.url = url
        self.title = title
        self.text = text
        # Pre-compute tokens for searching (title weighted 3x over body text)
        self._title_tokens = _tokenize(title)
        self._text_tokens = _tokenize(text)
        self.tf = _compute_tf(self._title_tokens * 3 + self._text_tokens)

    def as_dict(self) -> dict[str, str]:
        """Return a dictionary representation of the page."""
        return {
            "url": self.url,
            "title": self.title,
            "description": self.text,
        }


class _DocsIndex:
    """In-memory index of DataRobot documentation pages."""

    def __init__(self) -> None:
        self.pages: list[DocPage] = []
        self._df: dict[str, int] = {}  # document frequency
        self._built_at: float = 0.0  # timestamp of when the last index was built

    @property
    def is_stale(self) -> bool:
        """Check if the index needs refreshing."""
        if not self.pages:
            return True
        return (time.time() - self._built_at) > CACHE_TTL_SECONDS

    def build(self, pages: list[DocPage]) -> None:
        """
        Build the index from a list of pages.
        Computes document frequency of each term in the corpus.
        """
        self.pages = pages
        self._df = {}
        for page in pages:
            seen_terms: set[str] = set()
            for term in page.tf:
                if term not in seen_terms:
                    self._df[term] = self._df.get(term, 0) + 1
                    seen_terms.add(term)
        self._built_at = time.time()
        logger.info(f"Built docs index with {len(pages)} pages")

    def search(self, query: str, max_results: int = MAX_RESULTS_DEFAULT) -> list[DocPage]:
        """Search the index using TF-IDF scoring.

        Args:
            query: The search query string.
            max_results: Maximum number of results to return.

        Returns
        -------
            List of DocPage objects sorted by relevance score.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        num_docs = len(self.pages) if self.pages else 1
        scored: list[tuple[float, DocPage]] = []

        for page in self.pages:
            score = 0.0
            for token in query_tokens:
                tf = page.tf.get(token, 0.0)
                df = self._df.get(token, 0)
                if tf > 0 and df > 0:
                    idf = math.log(num_docs / df)
                    score += tf * idf
            if score > 0:
                scored.append((score, page))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [page for _, page in scored[:max_results]]


# Global index instance (cached)
_index = _DocsIndex()


async def _fetch_url_raw_text_content(session: aiohttp.ClientSession, url: str) -> str | None:
    """Fetch a URL and return its text content, or None on failure."""
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
            headers={"User-Agent": "DataRobot-Docs-Search/1.0"},
        ) as response:
            if response.status == 200:
                return await response.text()
            logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
            return None
    except Exception as e:
        logger.warning(f"Error fetching {url}: {e}")
        return None


async def _fetch_sitemap_urls(session: aiohttp.ClientSession) -> list[str]:
    """Fetch and parse the sitemap.xml to get agentic-ai documentation page URLs."""
    xml_text = await _fetch_url_raw_text_content(session, DOCS_SITEMAP_URL)
    if not xml_text:
        logger.warning("Sitemap not available; docs search will be empty until next refresh")
        return []

    urls: list[str] = []
    try:
        root = ElementTree.fromstring(xml_text)
        # Handle XML namespaces - sitemaps use the sitemap protocol namespace
        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for url_elem in root.findall(".//sm:url/sm:loc", ns):
            if url_elem.text:
                urls.append(url_elem.text.strip())
        # If no namespace, try without
        if not urls:
            for url_elem in root.findall(".//url/loc"):
                if url_elem.text:
                    urls.append(url_elem.text.strip())
    except ElementTree.ParseError as e:
        logger.warning(f"Failed to parse sitemap XML: {e}")
        return []

    # Filter to only agentic-ai pages
    docs_urls = [u for u in urls if AGENTIC_AI_PATH in u]
    logger.info(f"Found {len(docs_urls)} agentic-ai documentation URLs from sitemap")
    return docs_urls


def _title_from_url(url: str) -> str:
    """Generate a human-readable title from a URL path."""
    # Extract the path after /en/docs/
    path = url.split("/en/docs/")[-1] if "/en/docs/" in url else url
    # Remove file extensions and trailing slashes
    path = re.sub(r"\.(html|htm)$", "", path).strip("/")
    if not path:
        return "DataRobot Documentation Home"
    # Convert path segments to title
    segments = path.split("/")
    # Take the last meaningful segment and convert to title
    title_parts = []
    for segment in segments:
        words = re.sub(r"[-_]", " ", segment).strip()
        if words:
            title_parts.append(words.title())
    return " > ".join(title_parts) if title_parts else "DataRobot Documentation"


async def _build_index(session: aiohttp.ClientSession) -> list[DocPage]:
    """Build the documentation index from the sitemap with full page content.

    Fetches all agentic-ai documentation URLs from sitemap.xml, then retrieves
    their full HTML content concurrently. With ~28 pages and a concurrency limit
    of 30, this ideally should complete in a few seconds and provide TF-IDF scoring from
    real page titles and body text.
    """
    sitemap_urls = await _fetch_sitemap_urls(session)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_urls: list[str] = []
    for url in sitemap_urls:
        normalized = url.rstrip("/")
        if normalized not in seen:
            seen.add(normalized)
            unique_urls.append(url)

    logger.info(f"Fetching content for {len(unique_urls)} agentic-ai docs pages...")
    semaphore = asyncio.Semaphore(CONTENT_FETCH_CONCURRENCY)

    async def create_doc_page(url: str) -> DocPage:
        async with semaphore:
            html = await _fetch_url_raw_text_content(session, url)
        if html:
            extractor = _ContentExtractor()
            extractor.feed(html)
            title = extractor.title or _title_from_url(url)
            text = extractor.text_content
        else:
            title = _title_from_url(url)
            text = ""
        return DocPage(url=url, title=title, text=text)

    results = await asyncio.gather(
        *[create_doc_page(url) for url in unique_urls],
        return_exceptions=True,
    )
    pages = [r for r in results if isinstance(r, DocPage)]
    logger.info(f"Indexed {len(pages)} agentic-ai docs pages with full content")
    return pages


async def _ensure_index() -> _DocsIndex:
    """Ensure the docs index is built and up-to-date."""
    if not _index.is_stale:
        return _index

    logger.info("Building DataRobot agentic-ai docs index...")
    async with aiohttp.ClientSession() as session:
        pages = await _build_index(session)
        if pages:
            _index.build(pages)
        else:
            logger.warning("No pages found for docs index")

    return _index


async def search_docs(query: str, max_results: int = MAX_RESULTS_DEFAULT) -> list[dict[str, Any]]:
    """Search DataRobot agentic-ai documentation for pages relevant to a query.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.

    Returns
    -------
        List of dictionaries with 'url', 'title', and 'description' keys.
    """
    max_results = max(1, min(max_results, MAX_RESULTS))

    index = await _ensure_index()
    results = index.search(query, max_results=max_results)

    return [page.as_dict() for page in results]


async def fetch_page_content(url: str) -> dict[str, Any]:
    """Fetch and extract the text content of a specific documentation page.

    Args:
        url: The URL of the documentation page to fetch.

    Returns
    -------
        Dictionary with 'url', 'title', and 'content' keys.
    """
    if "/en/docs/" not in url or not url.startswith(DOCS_BASE_URL):
        return {
            "url": url,
            "title": "Error",
            "content": "URL must be a DataRobot English documentation page "
            f"(must contain '/en/docs/' or start with '{DOCS_BASE_URL}').",
        }

    # Check if the page is already cached in the index to avoid a live HTTP request
    if not _index.is_stale:
        for page in _index.pages:
            if page.url.lower().rstrip("/") == url.lower().rstrip("/"):
                return {
                    "url": page.url,
                    "title": page.title,
                    "content": page.text,
                }

    async with aiohttp.ClientSession() as session:
        html = await _fetch_url_raw_text_content(session, url)

    if not html:
        return {
            "url": url,
            "title": "Error",
            "content": f"Failed to fetch content from {url}",
        }

    extractor = _ContentExtractor()
    extractor.feed(html)

    title = extractor.title or _title_from_url(url)
    content = extractor.text_content

    return {
        "url": url,
        "title": title,
        "content": content,
    }
