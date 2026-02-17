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
from datarobot_genai.drmcp.tools.clients.dr_docs import _ContentExtractor
from datarobot_genai.drmcp.tools.clients.dr_docs import _DocsIndex
from datarobot_genai.drmcp.tools.clients.dr_docs import _title_from_url
from datarobot_genai.drmcp.tools.clients.dr_docs import _tokenize
from datarobot_genai.drmcp.tools.clients.dr_docs import fetch_page_content
from datarobot_genai.drmcp.tools.clients.dr_docs import search_docs


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


class TestContentExtractor:
    """Tests for _ContentExtractor HTML parser."""

    def test_extract_title(self) -> None:
        """Test extracting page title."""
        html = "<html><head><title>Test Page</title></head><body></body></html>"
        extractor = _ContentExtractor()
        extractor.feed(html)
        assert extractor.title == "Test Page"

    def test_extract_paragraph_text(self) -> None:
        """Test extracting paragraph text."""
        html = "<html><body><p>First paragraph</p><p>Second paragraph</p></body></html>"
        extractor = _ContentExtractor()
        extractor.feed(html)
        assert "First paragraph" in extractor.text_content
        assert "Second paragraph" in extractor.text_content

    def test_extract_heading_text(self) -> None:
        """Test extracting heading text."""
        html = "<html><body><h1>Main Title</h1><h2>Subtitle</h2></body></html>"
        extractor = _ContentExtractor()
        extractor.feed(html)
        content = extractor.text_content
        assert "Main Title" in content
        assert "Subtitle" in content

    def test_skip_script_tags(self) -> None:
        """Test that script content is skipped."""
        html = "<html><body><p>Visible</p><script>alert('hidden');</script></body></html>"
        extractor = _ContentExtractor()
        extractor.feed(html)
        content = extractor.text_content
        assert "Visible" in content
        assert "alert" not in content
        assert "hidden" not in content

    def test_skip_style_tags(self) -> None:
        """Test that style content is skipped."""
        html = "<html><body><p>Visible</p><style>body { color: red; }</style></body></html>"
        extractor = _ContentExtractor()
        extractor.feed(html)
        content = extractor.text_content
        assert "Visible" in content
        assert "color" not in content

    def test_skip_nav_tags(self) -> None:
        """Test that nav content is skipped."""
        html = "<html><body><p>Main</p><nav><a>Navigation</a></nav></body></html>"
        extractor = _ContentExtractor()
        extractor.feed(html)
        content = extractor.text_content
        assert "Main" in content
        assert "Navigation" not in content

    def test_extract_list_items(self) -> None:
        """Test extracting list item text."""
        html = "<html><body><ul><li>Item 1</li><li>Item 2</li></ul></body></html>"
        extractor = _ContentExtractor()
        extractor.feed(html)
        content = extractor.text_content
        assert "Item 1" in content
        assert "Item 2" in content

    def test_extract_table_content(self) -> None:
        """Test extracting table content."""
        html = "<html><body><table><tr><th>Header</th><td>Data</td></tr></table></body></html>"
        extractor = _ContentExtractor()
        extractor.feed(html)
        content = extractor.text_content
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
            url="https://docs.datarobot.com/en/docs/modeling/autopilot/",
            title="Autopilot Overview",
            text="Autopilot is an automated machine learning feature.",
        )
        assert page.url == "https://docs.datarobot.com/en/docs/modeling/autopilot/"
        assert page.title == "Autopilot Overview"
        assert "automated" in page.text

    def test_docpage_computes_tf(self) -> None:
        """Test that DocPage computes TF on initialization."""
        page = DocPage(
            url="https://docs.datarobot.com/en/docs/test/",
            title="Test Page",
            text="test content",
        )
        assert page.tf is not None
        assert isinstance(page.tf, dict)
        # Title tokens are weighted 3x, so "test" and "page" should appear
        assert "test" in page.tf
        assert "page" in page.tf

    def test_docpage_url_tokens_extracted(self) -> None:
        """Test that URL path is tokenized."""
        page = DocPage(
            url="https://docs.datarobot.com/en/docs/modeling/autopilot/",
            title="Title",
            text="",
        )
        # URL tokens should include "modeling" and "autopilot"
        assert "modeling" in page.tf
        assert "autopilot" in page.tf

    def test_docpage_title_weighted_higher(self) -> None:
        """Test that title tokens have higher weight than text tokens."""
        page = DocPage(
            url="https://docs.datarobot.com/en/docs/test/",
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
            url="https://docs.datarobot.com/en/docs/test/",
            title="Test",
            text="Short text",
        )
        result = page.as_dict()
        assert result["url"] == "https://docs.datarobot.com/en/docs/test/"
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


class TestSearchDocs:
    """Tests for search_docs function."""

    async def test_search_docs_builds_index(self) -> None:
        """Test that search_docs builds index if needed."""
        mock_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://docs.datarobot.com/en/docs/modeling/</loc></url>
        </urlset>"""

        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = mock_sitemap

            results = await search_docs("modeling", max_results=5)

            assert isinstance(results, list)
            # Should have called fetch for sitemap
            mock_fetch.assert_called()

    async def test_search_docs_returns_dicts(self) -> None:
        """Test that search_docs returns list of dicts."""
        mock_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://docs.datarobot.com/en/docs/modeling/test/</loc></url>
            <url><loc>https://docs.datarobot.com/en/docs/deployment/guide/</loc></url>
            <url><loc>https://docs.datarobot.com/en/docs/data/preparation/</loc></url>
        </urlset>"""

        # Force index rebuild by marking it as stale
        import datarobot_genai.drmcp.tools.clients.dr_docs as dr_docs_module

        dr_docs_module._index._built_at = 0

        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = mock_sitemap

            results = await search_docs("test", max_results=5)

            assert isinstance(results, list)
            assert len(results) > 0
            assert "url" in results[0]
            assert "title" in results[0]
            assert "description" in results[0]

    async def test_search_docs_clamps_max_results(self) -> None:
        """Test that max_results is clamped to valid range."""
        mock_sitemap = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://docs.datarobot.com/en/docs/test/</loc></url>
        </urlset>"""

        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = mock_sitemap

            # Test upper bound
            results = await search_docs("test", max_results=100)
            assert len(results) <= 20  # MAX_RESULTS = 20

            # Test lower bound - should not fail with max_results=0
            results = await search_docs("test", max_results=0)
            assert isinstance(results, list)


class TestFetchPageContent:
    """Tests for fetch_page_content function."""

    async def test_fetch_page_content_success(self) -> None:
        """Test fetching page content successfully."""
        mock_html = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Main Heading</h1>
                <p>This is test content.</p>
            </body>
        </html>
        """

        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url",
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
        assert "must be a DataRobot documentation page" in result["content"]

    async def test_fetch_page_content_fetch_failure(self) -> None:
        """Test handling fetch failure."""
        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url",
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
            <body><p>Content without title tag</p></body>
        </html>
        """

        with patch(
            "datarobot_genai.drmcp.tools.clients.dr_docs._fetch_url",
            new_callable=AsyncMock,
        ) as mock_fetch:
            mock_fetch.return_value = mock_html

            result = await fetch_page_content(
                "https://docs.datarobot.com/en/docs/modeling/autopilot/"
            )

            # Should use URL-derived title
            assert "Modeling" in result["title"]
            assert "Autopilot" in result["title"]
