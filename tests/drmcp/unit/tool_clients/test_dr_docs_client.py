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

from unittest.mock import AsyncMock
from unittest.mock import patch

from datarobot_genai.drmcp.tools.clients.dr_docs import DocPage
from datarobot_genai.drmcp.tools.clients.dr_docs import _compute_tf
from datarobot_genai.drmcp.tools.clients.dr_docs import _DocsIndex
from datarobot_genai.drmcp.tools.clients.dr_docs import _extract_html_content
from datarobot_genai.drmcp.tools.clients.dr_docs import _fetch_sitemap_urls
from datarobot_genai.drmcp.tools.clients.dr_docs import _title_from_url
from datarobot_genai.drmcp.tools.clients.dr_docs import _tokenize
from datarobot_genai.drmcp.tools.clients.dr_docs import fetch_page_content
from datarobot_genai.drmcp.tools.clients.dr_docs import search_docs

# Reusable agentic-ai mock sitemaps (must contain AGENTIC_AI_PATH to pass the filter)
_SITEMAP_ONE_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://docs.datarobot.com/en/docs/agentic-ai/index.html</loc></url>
</urlset>"""

_SITEMAP_TWO_PAGES = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://docs.datarobot.com/en/docs/agentic-ai/index.html</loc></url>
    <url><loc>https://docs.datarobot.com/en/docs/agentic-ai/agentic-glossary.html</loc></url>
</urlset>"""

_SITEMAP_THREE_PAGES = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
    <url><loc>https://docs.datarobot.com/en/docs/agentic-ai/index.html</loc></url>
    <url><loc>https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/index.html</loc></url>
    <url><loc>https://docs.datarobot.com/en/docs/agentic-ai/agentic-deploy/index.html</loc></url>
</urlset>"""


class TestFetchSitemapUrls:
    """Tests for _fetch_sitemap_urls."""

    async def test_fetch_sitemap_urls_namespaced(self) -> None:
        """Parses a standard namespaced sitemap."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url_raw_text_content",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = _SITEMAP_TWO_PAGES

            urls = await _fetch_sitemap_urls(session=object())  # type: ignore[arg-type]

        assert urls == [
            "https://docs.datarobot.com/en/docs/agentic-ai/index.html",
            "https://docs.datarobot.com/en/docs/agentic-ai/agentic-glossary.html",
        ]

    async def test_fetch_sitemap_urls_without_namespace_filters_agentic_ai(self) -> None:
        """Falls back to non-namespaced sitemap parsing and filters to agentic-ai."""
        sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset>
            <url><loc>https://docs.datarobot.com/en/docs/agentic-ai/index.html</loc></url>
            <url><loc>https://docs.datarobot.com/en/docs/modeling/index.html</loc></url>
        </urlset>"""

        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url_raw_text_content",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = sitemap

            urls = await _fetch_sitemap_urls(session=object())  # type: ignore[arg-type]

        assert urls == ["https://docs.datarobot.com/en/docs/agentic-ai/index.html"]


class TestTokenize:
    """Tests for _tokenize function."""

    def test_tokenize_basic_text(self) -> None:
        """Test tokenizing simple text."""
        result = _tokenize("Hello World")
        assert result == ["hello", "world"]

    def test_tokenize_with_numbers(self) -> None:
        """Test tokenizing text with numbers."""
        result = _tokenize("Python 3.11 is great")
        assert result == ["python", "3", "11", "is", "great"]

    def test_tokenize_strips_punctuation(self) -> None:
        """Test that punctuation is stripped."""
        result = _tokenize("Hello, World! How are you?")
        assert result == ["hello", "world", "how", "are", "you"]

    def test_tokenize_lowercases(self) -> None:
        """Test that text is lowercased."""
        result = _tokenize("UPPERCASE MixedCase")
        assert result == ["uppercase", "mixedcase"]

    def test_tokenize_empty_string(self) -> None:
        """Test tokenizing empty string."""
        result = _tokenize("")
        assert result == []

    def test_tokenize_only_punctuation(self) -> None:
        """Test tokenizing string with only punctuation."""
        result = _tokenize("!@#$%^&*()")
        assert result == []


class TestComputeTf:
    """Tests for _compute_tf function."""

    def test_compute_tf_basic(self) -> None:
        """Test computing term frequency for simple list."""
        tokens = ["model", "model", "deployment"]
        result = _compute_tf(tokens)
        assert result == {"model": 2 / 3, "deployment": 1 / 3}

    def test_compute_tf_single_token(self) -> None:
        """Test computing TF for single token."""
        tokens = ["model"]
        result = _compute_tf(tokens)
        assert result == {"model": 1.0}

    def test_compute_tf_empty_list(self) -> None:
        """Test computing TF for empty list."""
        result = _compute_tf([])
        assert result == {}

    def test_compute_tf_all_same(self) -> None:
        """Test computing TF when all tokens are the same."""
        tokens = ["test", "test", "test"]
        result = _compute_tf(tokens)
        assert result == {"test": 1.0}


class TestExtractHtmlContent:
    """Tests for _extract_html_content."""

    def test_extract_paragraph_text(self) -> None:
        """Test extracting paragraph text."""
        html = """
        <html>
          <head>
            <title>Paragraphs</title>
          </head>
          <body>
            <p>First paragraph</p>
            <p>Second paragraph</p>
          </body>
        </html>
        """
        title, content = _extract_html_content(html)
        assert title == "Paragraphs"
        assert "First paragraph" in content
        assert "Second paragraph" in content

    def test_extract_heading_text(self) -> None:
        """Test extracting heading text."""
        html = """
        <html>
          <head>
            <title>Headings</title>
          </head>
          <body>
            <h1>Main Title</h1>
            <h2>Subtitle</h2>
          </body>
        </html>
        """
        title, content = _extract_html_content(html)
        assert title == "Headings"
        assert "Main Title" in content
        assert "Subtitle" in content

    def test_skip_script_tags(self) -> None:
        """Test that script content is skipped."""
        html = """
        <html>
          <head>
            <title>Scripts</title>
          </head>
          <body>
            <p>Visible</p>
            <script>alert('hidden');</script>
          </body>
        </html>
        """
        title, content = _extract_html_content(html)
        assert title == "Scripts"
        assert "Visible" in content
        assert "alert" not in content
        assert "hidden" not in content

    def test_skip_style_tags(self) -> None:
        """Test that style content is skipped."""
        html = """
        <html>
          <head>
            <title>Styles</title>
          </head>
          <body>
            <p>Visible</p>
            <style>body { color: red; }</style>
          </body>
        </html>
        """
        title, content = _extract_html_content(html)
        assert title == "Styles"
        assert "Visible" in content
        assert "color" not in content

    def test_skip_nav_tags(self) -> None:
        """Test that nav content is skipped."""
        html = """
        <html>
          <head>
            <title>Navigation</title>
          </head>
          <body>
            <p>Main</p>
            <nav><a>Nav Link</a></nav>
          </body>
        </html>
        """
        title, content = _extract_html_content(html)
        assert title == "Navigation"
        assert "Main" in content
        assert "Nav Link" not in content

    def test_extract_list_items(self) -> None:
        """Test extracting list item text."""
        html = """
        <html>
          <head>
            <title>List</title>
          </head>
          <body>
            <ul>
              <li>Item 1</li>
              <li>Item 2</li>
            </ul>
          </body>
        </html>
        """
        title, content = _extract_html_content(html)
        assert title == "List"
        assert "Item 1" in content
        assert "Item 2" in content

    def test_extract_table_content(self) -> None:
        """Test extracting table content."""
        html = """
        <html>
          <head>
            <title>Table</title>
          </head>
          <body>
            <table>
              <tr>
                <th>Header</th>
                <td>Data</td>
              </tr>
            </table>
          </body>
        </html>
        """
        title, content = _extract_html_content(html)
        assert title == "Table"
        assert "Header" in content
        assert "Data" in content


class TestTitleFromUrl:
    """Tests for _title_from_url function."""

    def test_title_from_simple_path(self) -> None:
        """Test generating title from simple path."""
        url = "https://docs.datarobot.com/en/docs/modeling/autopilot/"
        result = _title_from_url(url)
        assert result == "Modeling > Autopilot"

    def test_title_from_path_with_hyphens(self) -> None:
        """Test generating title from path with hyphens."""
        url = "https://docs.datarobot.com/en/docs/data/feature-discovery/"
        result = _title_from_url(url)
        assert result == "Data > Feature Discovery"

    def test_title_from_path_with_underscores(self) -> None:
        """Test generating title from path with underscores."""
        url = "https://docs.datarobot.com/en/docs/api/api_reference/"
        result = _title_from_url(url)
        assert result == "Api > Api Reference"

    def test_title_from_homepage(self) -> None:
        """Test generating title for homepage."""
        url = "https://docs.datarobot.com/en/docs/"
        result = _title_from_url(url)
        assert result == "DataRobot Documentation Home"

    def test_title_removes_html_extension(self) -> None:
        """Test that .html extension is removed."""
        url = "https://docs.datarobot.com/en/docs/modeling/index.html"
        result = _title_from_url(url)
        assert result == "Modeling > Index"
        assert ".html" not in result


class TestDocPage:
    """Tests for DocPage class."""

    def test_docpage_initialization(self) -> None:
        """Test DocPage initializes correctly."""
        page = DocPage(
            url="https://docs.datarobot.com/en/docs/agentic-ai/index.html",
            title="Agentic AI Overview",
            text="Agentic AI is a feature.",
        )
        assert page.url == "https://docs.datarobot.com/en/docs/agentic-ai/index.html"
        assert page.title == "Agentic AI Overview"
        assert "feature" in page.text

    def test_docpage_computes_tf(self) -> None:
        """Test that DocPage computes TF on initialization."""
        page = DocPage(
            url="https://docs.datarobot.com/en/docs/agentic-ai/test/",
            title="Test Page",
            text="test content",
        )
        assert page.tf is not None
        assert isinstance(page.tf, dict)
        # Title tokens are weighted 3x, so "test" and "page" should appear
        assert "test" in page.tf
        assert "page" in page.tf

    def test_docpage_title_tokens_extracted(self) -> None:
        """Test that title tokens are indexed for search."""
        page = DocPage(
            url="https://docs.datarobot.com/en/docs/agentic-ai/agentic-glossary.html",
            title="Agentic AI > Agentic Glossary",
            text="sample text",
        )
        # Tokens should include "agentic", "glossary", "sample"
        assert "agentic" in page.tf
        assert "glossary" in page.tf
        assert "sample" in page.tf

    def test_docpage_title_weighted_higher(self) -> None:
        """Test that title tokens have higher weight than text tokens."""
        page = DocPage(
            url="https://docs.datarobot.com/en/docs/agentic-ai/test/",
            title="unique",
            text="common common common",
        )
        # "unique" appears 3 times (title weight), "common" appears 3 times (text weight)
        # Total tokens: 6, so both should have same count but title was repeated 3x
        tf_unique = page.tf.get("unique")
        tf_common = page.tf.get("common")
        assert tf_unique == tf_common  # Both appear same number of times in weighted list

    def test_docpage_as_dict(self) -> None:
        """Test DocPage.as_dict() returns correct format."""
        page = DocPage(
            url="https://docs.datarobot.com/en/docs/agentic-ai/test/",
            title="Test",
            text="Short text",
        )
        result = page.as_dict()
        assert result["url"] == "https://docs.datarobot.com/en/docs/agentic-ai/test/"
        assert result["title"] == "Test"
        assert result["description"] == "Short text"


class TestDocsIndex:
    """Tests for _DocsIndex class."""

    def test_index_initialization(self) -> None:
        """Test index initializes empty."""
        index = _DocsIndex()
        assert index.pages == []
        assert index.is_stale is True

    def test_index_build(self) -> None:
        """Test building the index."""
        pages = [
            DocPage("https://example.com/page1", "Page 1", "content one"),
            DocPage("https://example.com/page2", "Page 2", "content two"),
        ]
        index = _DocsIndex()
        index.build(pages)
        assert len(index.pages) == 2
        assert index.is_stale is False

    def test_index_df_computed(self) -> None:
        """Test that document frequency is computed correctly."""
        pages = [
            DocPage("https://example.com/page1", "model training", ""),
            DocPage("https://example.com/page2", "model deployment", ""),
            DocPage("https://example.com/page3", "data preparation", ""),
        ]
        index = _DocsIndex()
        index.build(pages)
        # "model" appears in 2 docs, "training" in 1, "deployment" in 1, "data" in 1
        assert index._df["model"] == 2
        assert index._df["training"] == 1
        assert index._df["deployment"] == 1
        assert index._df["data"] == 1

    def test_index_search_returns_relevant_pages(self) -> None:
        """Test that search returns relevant pages."""
        pages = [
            DocPage("https://example.com/model", "Model Training", "train models"),
            DocPage("https://example.com/data", "Data Preparation", "prepare data"),
            DocPage("https://example.com/deploy", "Deployment Guide", "deploy models"),
        ]
        index = _DocsIndex()
        index.build(pages)
        results = index.search("model", max_results=5)
        # Should return pages with "model" in title/url
        assert 0 < len(results) <= 5
        assert any("model" in p.url.lower() for p in results)

    def test_index_search_empty_query(self) -> None:
        """Test that empty query returns no results."""
        pages = [DocPage("https://example.com/page", "Page", "content")]
        index = _DocsIndex()
        index.build(pages)
        results = index.search("", max_results=5)
        assert results == []

    def test_index_search_no_matches(self) -> None:
        """Test that search with no matches returns empty list."""
        pages = [DocPage("https://example.com/page", "Page Title", "some content")]
        index = _DocsIndex()
        index.build(pages)
        results = index.search("nonexistent", max_results=5)
        assert results == []

    def test_index_search_tfidf_scoring(self) -> None:
        """Test that TF-IDF scoring works correctly."""
        pages = [
            # Page with rare term "autopilot" should score higher
            DocPage("https://example.com/autopilot", "Autopilot", "autopilot features"),
            # Page with common term "the" should score lower
            DocPage("https://example.com/common", "Common Words", "the the the"),
        ]
        index = _DocsIndex()
        index.build(pages)
        results = index.search("autopilot", max_results=2)
        assert len(results) == 1  # Only autopilot page matches
        assert "autopilot" in results[0].url


class TestSearchAgenticDocs:
    """Tests for search_docs function."""

    async def test_search_agentic_docs_builds_index(self) -> None:
        """Test that search_docs builds index if needed."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url_raw_text_content",
            new_callable=AsyncMock,
        ) as mock_fetch:

            async def _side_effect(session: object, url: str) -> str:
                if "sitemap" in url:
                    return _SITEMAP_ONE_PAGE
                return """
                <html>
                  <head>
                    <title>Test Page</title>
                  </head>
                  <body>
                    <p>Test content</p>
                  </body>
                </html>
                """

            mock_fetch.side_effect = _side_effect

            results = await search_docs("agentic", max_results=5)

            assert isinstance(results, list)
            # Should have called fetch for sitemap + page content
            mock_fetch.assert_called()

    async def test_search_agentic_docs_returns_dicts(self) -> None:
        """Test that search_docs returns list of dicts."""
        import datarobot_genai.drmcp.tools.clients.dr_docs as dr_docs_module

        # Force index rebuild by marking it as stale
        dr_docs_module._index._built_at = 0

        def _html(title: str, body: str) -> str:
            return f"<html><head><title>{title}</title></head><body><p>{body}</p></body></html>"

        # Give each page distinct content so TF-IDF can differentiate them
        page_html = {
            "https://docs.datarobot.com/en/docs/agentic-ai/index.html": _html(
                "Agentic AI Overview", "Introduction to agentic AI features"
            ),
            "https://docs.datarobot.com/en/docs/agentic-ai/agentic-develop/index.html": _html(
                "Agentic Development Guide", "How to develop agentic workflows"
            ),
            "https://docs.datarobot.com/en/docs/agentic-ai/agentic-deploy/index.html": _html(
                "Agentic Deployment Guide", "How to deploy agentic applications"
            ),
        }

        async def _side_effect(session: object, url: str) -> str:
            if "sitemap" in url:
                return _SITEMAP_THREE_PAGES
            return page_html.get(url)

        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url_raw_text_content",
            side_effect=_side_effect,
        ):
            # "develop" only appears in one page → non-zero IDF → at least one result
            results = await search_docs("develop", max_results=5)

            assert isinstance(results, list)
            assert len(results) > 0
            assert "url" in results[0]
            assert "title" in results[0]
            assert "description" in results[0]

    async def test_search_agentic_docs_clamps_max_results(self) -> None:
        """Test that max_results is clamped to valid range."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url_raw_text_content",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = _SITEMAP_ONE_PAGE

            # Test upper bound
            results = await search_docs("agentic", max_results=100)
            assert len(results) <= 10  # MAX_RESULTS = 10

            # Test lower bound - should not fail with max_results=0
            results = await search_docs("agentic", max_results=0)
            assert isinstance(results, list)


class TestFetchPageContent:
    """Tests for fetch_page_content function."""

    async def test_fetch_page_content_success(self) -> None:
        """Test fetching page content successfully."""
        mock_html = """
        <html>
          <head>
            <title>Test Page</title>
          </head>
          <body>
            <h1>Main Heading</h1>
            <p>This is test content.</p>
          </body>
        </html>
        """

        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url_raw_text_content",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = mock_html

            result = await fetch_page_content("https://docs.datarobot.com/en/docs/test/")

            assert result["url"] == "https://docs.datarobot.com/en/docs/test/"
            assert result["title"] == "Test Page"
            assert "Main Heading" in result["content"]
            assert "This is test content" in result["content"]

    async def test_fetch_page_content_invalid_url(self) -> None:
        """Test fetching page with invalid URL returns error."""
        result = await fetch_page_content("https://example.com/not-docs/")

        assert result["title"] == "Error"
        assert "must be a DataRobot English documentation page" in result["content"]

    async def test_fetch_page_content_fetch_failure(self) -> None:
        """Test handling fetch failure."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url_raw_text_content",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = None

            result = await fetch_page_content("https://docs.datarobot.com/en/docs/test/")

            assert result["title"] == "Error"
            assert "Failed to fetch content" in result["content"]

    async def test_fetch_page_content_uses_url_title_fallback(self) -> None:
        """Test that URL-based title is used when <title> tag missing."""
        mock_html = """
        <html>
          <body>
            <p>Content without title tag</p>
          </body>
        </html>
        """

        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url_raw_text_content",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = mock_html

            result = await fetch_page_content(
                "https://docs.datarobot.com/en/docs/modeling/autopilot/"
            )

            # Should use URL-derived title
            assert "Modeling" in result["title"]
            assert "Autopilot" in result["title"]
