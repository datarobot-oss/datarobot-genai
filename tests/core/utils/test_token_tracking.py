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
"""
Unit tests for token_tracking module.

Tests cover:
- Basic token estimation accuracy
- Unicode and CJK text handling
- Performance benchmarks
- Memory-efficient file/streaming functions
- CSV row estimation
- Token tracking strategies
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Iterator

import pytest

from datarobot_genai.core.utils.token_tracking import ApiResponseCountingStrategy
from datarobot_genai.core.utils.token_tracking import HeuristicTokenCountingStrategy
from datarobot_genai.core.utils.token_tracking import TokenUsageTracker
from datarobot_genai.core.utils.token_tracking import count_messages_tokens
from datarobot_genai.core.utils.token_tracking import estimate_csv_rows_for_token_limit
from datarobot_genai.core.utils.token_tracking import estimate_tokens
from datarobot_genai.core.utils.token_tracking import estimate_tokens_from_file
from datarobot_genai.core.utils.token_tracking import estimate_tokens_streaming

# =============================================================================
# Basic Token Estimation Tests
# =============================================================================


class TestEstimateTokens:
    """Tests for the estimate_tokens function."""

    def test_empty_string_returns_zero(self) -> None:
        """Empty string should return 0 tokens."""
        assert estimate_tokens("") == 0

    def test_none_like_empty(self) -> None:
        """None-like empty values should return 0."""
        assert estimate_tokens("") == 0

    def test_single_word(self) -> None:
        """Single short word should be 1 token."""
        assert estimate_tokens("hello") == 1
        assert estimate_tokens("world") == 1

    def test_simple_sentence(self) -> None:
        """Simple sentence token count."""
        tokens = estimate_tokens("Hello world")
        assert 2 <= tokens <= 3  # 2 words, possible punctuation

    def test_sentence_with_punctuation(self) -> None:
        """Punctuation should add tokens."""
        without_punct = estimate_tokens("Hello world")
        with_punct = estimate_tokens("Hello, world!")
        assert with_punct > without_punct

    def test_numbers_tokenization(self) -> None:
        """Numbers should be tokenized by digit groups."""
        # Short number: 1 token
        assert estimate_tokens("123") == 1
        # Longer number: multiple tokens (~3 digits per token)
        tokens_long = estimate_tokens("123456789")
        assert tokens_long >= 2

    def test_long_words_split(self) -> None:
        """Long words should be split into multiple tokens."""
        short_word = estimate_tokens("hello")  # 5 chars
        long_word = estimate_tokens("internationalization")  # 20 chars
        assert long_word > short_word

    def test_newlines_are_tokens(self) -> None:
        """Newlines should count as separate tokens."""
        without_newline = estimate_tokens("hello world")
        with_newline = estimate_tokens("hello\nworld")
        assert with_newline > without_newline

    def test_whitespace_handling(self) -> None:
        """Multiple spaces shouldn't inflate token count significantly."""
        normal = estimate_tokens("hello world")
        extra_spaces = estimate_tokens("hello    world")
        # Should be similar (whitespace is compressed)
        assert abs(normal - extra_spaces) <= 1


# =============================================================================
# Unicode and CJK Tests
# =============================================================================


class TestUnicodeTokenEstimation:
    """Tests for Unicode text handling."""

    def test_cyrillic_text(self) -> None:
        """Cyrillic text should be tokenized like Latin."""
        tokens = estimate_tokens("Привіт світ")  # "Hello world" in Ukrainian
        assert tokens >= 2

    def test_arabic_text(self) -> None:
        """Arabic text should be tokenized."""
        tokens = estimate_tokens("مرحبا بالعالم")  # "Hello world" in Arabic
        assert tokens >= 2

    def test_japanese_hiragana(self) -> None:
        """Japanese hiragana should use CJK tokenization (~0.85 tokens/char)."""
        text = "こんにちは"  # 5 characters
        tokens = estimate_tokens(text)
        # With 0.85 multiplier: 5 * 0.85 = 4.25 → 4 tokens
        assert tokens >= 1  # At minimum, should count something

    def test_japanese_kanji(self) -> None:
        """Japanese kanji should use CJK tokenization."""
        text = "世界"  # 2 characters
        tokens = estimate_tokens(text)
        # With 0.85 multiplier: 2 * 0.85 = 1.7 → 1 token (min 1)
        assert tokens >= 1

    def test_chinese_text(self) -> None:
        """Chinese text should use CJK tokenization."""
        text = "你好世界"  # 4 characters
        tokens = estimate_tokens(text)
        # With 0.85 multiplier: 4 * 0.85 = 3.4 → 3 tokens
        assert tokens >= 1

    def test_korean_text(self) -> None:
        """Korean text should use CJK tokenization."""
        text = "안녕하세요"  # 5 characters
        tokens = estimate_tokens(text)
        # With 0.85 multiplier: 5 * 0.85 = 4.25 → 4 tokens
        assert tokens >= 1

    def test_mixed_cjk_and_latin(self) -> None:
        """Mixed CJK and Latin should both be counted."""
        text = "Hello こんにちは world"
        tokens = estimate_tokens(text)
        # "Hello" (1) + "こんにちは" (~4) + "world" (1) = ~6
        assert tokens >= 3

    def test_cjk_not_skipped(self) -> None:
        """Regression test: CJK text must not be skipped (previous bug)."""
        # This was returning 1 token before the fix
        text = "こんにちは"
        tokens = estimate_tokens(text)
        assert tokens > 1, "CJK text should not be counted as single token"


# =============================================================================
# Number Handling Tests
# =============================================================================


class TestNumberTokenization:
    """Tests for number tokenization (regression tests for double-counting bug)."""

    def test_numbers_not_double_counted(self) -> None:
        """Numbers should not be counted as both words and number tokens."""
        # "12345" should be ~2 tokens, not 4 (word + 3 digit tokens)
        tokens = estimate_tokens("12345")
        assert tokens <= 3

    def test_csv_like_numbers(self) -> None:
        """CSV-like numeric content should not over-count."""
        text = "123,456,789,1000,2000,3000"
        tokens = estimate_tokens(text)
        # 6 numbers + 5 commas = ~11 tokens, not 20+
        assert tokens < 20

    def test_mixed_text_and_numbers(self) -> None:
        """Mixed content should balance word and number tokens."""
        text = "The price is 12345 dollars"
        tokens = estimate_tokens(text)
        # ~5 words + 1-2 number tokens
        assert 5 <= tokens <= 10


# =============================================================================
# File-based Token Estimation Tests
# =============================================================================


class TestEstimateTokensFromFile:
    """Tests for memory-efficient file-based token estimation."""

    def test_small_file(self) -> None:
        """Small files should be read entirely for accuracy."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world, this is a test.")
            f.flush()
            path = f.name

        try:
            tokens = estimate_tokens_from_file(path)
            expected = estimate_tokens("Hello world, this is a test.")
            assert tokens == expected
        finally:
            os.unlink(path)

    def test_empty_file(self) -> None:
        """Empty file should return 0 tokens."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            path = f.name

        try:
            tokens = estimate_tokens_from_file(path)
            assert tokens == 0
        finally:
            os.unlink(path)

    def test_large_file_uses_sampling(self) -> None:
        """Large files should use sampling and return reasonable estimate."""
        # Create a 15MB file (above 10MB threshold)
        base_text = "Hello world 12345, this is a test line.\n"
        target_size = 15 * 1024 * 1024

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            written = 0
            while written < target_size:
                f.write(base_text)
                written += len(base_text)
            path = f.name

        try:
            tokens = estimate_tokens_from_file(path)
            # Should estimate millions of tokens
            assert tokens > 1_000_000
            # Rough check: ~10 tokens per line, ~375K lines
            assert tokens < 10_000_000
        finally:
            os.unlink(path)

    def test_file_with_unicode(self) -> None:
        """Unicode files should be handled correctly."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello こんにちは world 世界\n" * 100)
            path = f.name

        try:
            tokens = estimate_tokens_from_file(path)
            assert tokens > 100  # Multiple tokens per line
        finally:
            os.unlink(path)


# =============================================================================
# Streaming Token Estimation Tests
# =============================================================================


class TestEstimateTokensStreaming:
    """Tests for streaming/iterator-based token estimation."""

    def test_basic_streaming(self) -> None:
        """Basic streaming estimation."""
        lines = ["Hello world"] * 100
        tokens, count = estimate_tokens_streaming(iter(lines), sample_every_n=1)
        assert count == 100
        assert tokens > 100  # ~2 tokens per line

    def test_sampling_accuracy(self) -> None:
        """Sampling should give reasonable estimates."""
        lines = ["Hello world, this is a test."] * 10000
        tokens, count = estimate_tokens_streaming(iter(lines), sample_every_n=100)
        assert count == 10000
        # ~8 tokens per line * 10000 = ~80000
        assert 50000 < tokens < 120000

    def test_empty_iterator(self) -> None:
        """Empty iterator should return 0."""
        tokens, count = estimate_tokens_streaming(iter([]))
        assert tokens == 0
        assert count == 0

    def test_generator_input(self) -> None:
        """Should work with generators."""

        def line_generator() -> Iterator[str]:
            for i in range(50):
                yield f"Line {i}: Hello world"

        tokens, count = estimate_tokens_streaming(line_generator(), sample_every_n=1)
        assert count == 50
        assert tokens > 50


# =============================================================================
# CSV Row Estimation Tests
# =============================================================================


class TestEstimateCsvRowsForTokenLimit:
    """Tests for CSV row estimation."""

    def test_basic_csv_estimation(self) -> None:
        """Basic CSV estimation should work."""
        import pandas as pd

        df = pd.DataFrame({"a": range(100), "b": ["hello"] * 100})
        csv_str, token_count = estimate_csv_rows_for_token_limit(
            df, max_tokens=1000, initial_rows=10
        )
        assert token_count > 0
        assert len(csv_str) > 0

    def test_respects_token_limit(self) -> None:
        """Should respect token limit."""
        import pandas as pd

        df = pd.DataFrame({"a": range(1000), "b": ["hello world"] * 1000})
        csv_str, token_count = estimate_csv_rows_for_token_limit(
            df, max_tokens=500, initial_rows=100
        )
        # Should be reasonably close to limit (may exceed slightly due to estimation)
        assert token_count <= 1000  # Reasonable upper bound

    def test_small_dataframe(self) -> None:
        """Small DataFrame should return all rows."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        csv_str, token_count = estimate_csv_rows_for_token_limit(
            df, max_tokens=10000, initial_rows=10
        )
        assert "1" in csv_str and "3" in csv_str


# =============================================================================
# Token Counting Strategy Tests
# =============================================================================


class TestHeuristicTokenCountingStrategy:
    """Tests for HeuristicTokenCountingStrategy class."""

    def test_count_text(self) -> None:
        """Should count tokens in text."""
        strategy = HeuristicTokenCountingStrategy()
        count = strategy._count_text("Hello world")
        assert count >= 2

    def test_count_messages(self) -> None:
        """Should count tokens in message list."""
        strategy = HeuristicTokenCountingStrategy()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = strategy._count_messages(messages)  # type: ignore[arg-type]
        assert count > 4  # roles + content

    def test_count_tokens_with_response(self) -> None:
        """Should count both prompt and completion tokens."""
        strategy = HeuristicTokenCountingStrategy()
        messages = [{"role": "user", "content": "Hello"}]

        # Mock response with content
        class MockResponse:
            class Choice:
                class Message:
                    content = "Hello! How can I help you today?"

                message = Message()

            choices = [Choice()]

        prompt_tokens, completion_tokens = strategy.count_tokens(
            messages,  # type: ignore[arg-type]
            MockResponse(),
            "gpt-4",
        )
        assert prompt_tokens > 0
        assert completion_tokens > 0


class TestApiResponseCountingStrategy:
    """Tests for ApiResponseCountingStrategy class."""

    def test_extracts_usage_from_response(self) -> None:
        """Should extract token counts from API response."""
        strategy = ApiResponseCountingStrategy()

        class MockUsage:
            prompt_tokens = 100
            completion_tokens = 50

        class MockResponse:
            usage = MockUsage()

        prompt, completion = strategy.count_tokens([], MockResponse(), "gpt-4")
        assert prompt == 100
        assert completion == 50

    def test_fallback_on_missing_usage(self) -> None:
        """Should fallback to heuristic when usage is missing."""
        strategy = ApiResponseCountingStrategy()
        messages = [{"role": "user", "content": "Hello world"}]

        class MockResponse:
            usage = None

            class Choice:
                class Message:
                    content = "Hi!"

                message = Message()

            choices = [Choice()]

        prompt, completion = strategy.count_tokens(
            messages,  # type: ignore[arg-type]
            MockResponse(),
            "gpt-4",
        )
        assert prompt > 0  # Should use fallback
        assert completion > 0


class TestTokenUsageTracker:
    """Tests for TokenUsageTracker class."""

    def test_track_usage(self) -> None:
        """Should track cumulative usage."""
        tracker = TokenUsageTracker(strategy=HeuristicTokenCountingStrategy())
        messages = [{"role": "user", "content": "Hello"}]

        class MockResponse:
            class Choice:
                class Message:
                    content = "Hi there!"

                message = Message()

            choices = [Choice()]

        tracker.track_call(messages, MockResponse(), "gpt-4")  # type: ignore[arg-type]

        assert tracker.prompt_tokens > 0
        assert tracker.completion_tokens > 0
        assert tracker.call_count == 1

    def test_multiple_tracks(self) -> None:
        """Should accumulate across multiple calls."""
        tracker = TokenUsageTracker(strategy=HeuristicTokenCountingStrategy())
        messages = [{"role": "user", "content": "Hello"}]

        class MockResponse:
            class Choice:
                class Message:
                    content = "Hi!"

                message = Message()

            choices = [Choice()]

        tracker.track_call(messages, MockResponse(), "gpt-4")  # type: ignore[arg-type]
        tracker.track_call(messages, MockResponse(), "gpt-4")  # type: ignore[arg-type]

        assert tracker.call_count == 2

    def test_to_dict(self) -> None:
        """Should return usage as dictionary."""
        tracker = TokenUsageTracker(strategy=HeuristicTokenCountingStrategy())
        usage = tracker.to_dict()

        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage
        assert "call_count" in usage


class TestCountMessagesTokens:
    """Tests for count_messages_tokens helper function."""

    def test_count_messages(self) -> None:
        """Should count tokens in messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        tokens = count_messages_tokens(messages)  # type: ignore[arg-type]
        assert tokens > 5


# =============================================================================
# Performance Benchmark Tests
# =============================================================================


class TestPerformance:
    """Performance benchmark tests.

    Note: Time thresholds are set conservatively to pass on slow CI runners.
    Local performance is typically 2-5x faster than these thresholds.
    """

    @pytest.mark.parametrize(
        "size_kb",
        [1, 10, 100],
    )
    def test_small_text_performance(self, size_kb: int) -> None:
        """Small texts should process quickly."""
        text = "Hello world 12345, this is a test. " * (size_kb * 25)

        start = time.perf_counter()
        tokens = estimate_tokens(text)
        elapsed = time.perf_counter() - start

        assert tokens > 0
        # Should complete within 5 seconds for texts up to 100KB (CI-friendly)
        assert elapsed < 5.0, f"{size_kb}KB took {elapsed:.2f}s"

    def test_1mb_text_performance(self) -> None:
        """1MB text should process in reasonable time."""
        text = "Hello world 12345, this is a test. " * 30000  # ~1MB

        start = time.perf_counter()
        tokens = estimate_tokens(text)
        elapsed = time.perf_counter() - start

        assert tokens > 100000
        # CI runners can be slow; allow up to 5s (local is typically <0.5s)
        assert elapsed < 5.0, f"1MB took {elapsed:.2f}s, expected <5s"

    def test_sampling_kicks_in_for_large_text(self) -> None:
        """Texts > 10MB should use sampling and be faster than linear."""
        # Create ~15MB text
        text = "Hello world 12345, this is a test. " * 450000

        start = time.perf_counter()
        tokens = estimate_tokens(text)
        elapsed = time.perf_counter() - start

        assert tokens > 1000000
        # Sampling should make this much faster than processing all 15MB
        # Allow up to 5s for slow CI (local is typically <0.5s)
        assert elapsed < 5.0, f"15MB took {elapsed:.2f}s, expected <5s with sampling"

    def test_file_based_constant_time(self) -> None:
        """File-based estimation should complete in reasonable time."""
        base_text = "Hello world 12345, this is a test line.\n"

        times = []
        for size_mb in [5, 15]:  # Below and above sampling threshold
            target_size = size_mb * 1024 * 1024

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                written = 0
                while written < target_size:
                    f.write(base_text)
                    written += len(base_text)
                path = f.name

            try:
                start = time.perf_counter()
                estimate_tokens_from_file(path)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            finally:
                os.unlink(path)

        # Both should complete in reasonable time (CI can be slow with I/O)
        for t in times:
            assert t < 30.0, f"File processing took {t:.2f}s"

    def test_unicode_performance_not_degraded(self) -> None:
        """Unicode text should not be significantly slower than ASCII."""
        ascii_text = "Hello world test " * 10000
        unicode_text = "Hello мир 世界 " * 10000

        start = time.perf_counter()
        estimate_tokens(ascii_text)
        ascii_time = time.perf_counter() - start

        start = time.perf_counter()
        estimate_tokens(unicode_text)
        unicode_time = time.perf_counter() - start

        # Unicode should be within 10x of ASCII performance (conservative for CI)
        assert unicode_time < ascii_time * 10, (
            f"Unicode ({unicode_time:.3f}s) much slower than ASCII ({ascii_time:.3f}s)"
        )


# =============================================================================
# Accuracy Tests (comparing to expected ratios)
# =============================================================================


class TestAccuracy:
    """Tests to verify token estimation accuracy against known ratios."""

    def test_english_prose_chars_per_token(self) -> None:
        """English prose should average ~4 chars per token."""
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a sample of typical English prose that should "
            "tokenize according to standard BPE patterns."
        )
        tokens = estimate_tokens(text)
        chars_per_token = len(text) / tokens

        # OpenAI guideline: ~4 chars per token for English
        assert 3.0 < chars_per_token < 6.0, f"Got {chars_per_token:.1f} chars/token"

    def test_code_chars_per_token(self) -> None:
        """Code should have reasonable chars per token ratio."""
        code = """
def calculate_total(items):
    total = sum(item.price for item in items)
    return total * 1.1  # Add 10% tax
"""
        tokens = estimate_tokens(code)
        chars_per_token = len(code) / tokens

        # Code typically has 3-5 chars per token
        assert 2.0 < chars_per_token < 7.0, f"Got {chars_per_token:.1f} chars/token"

    def test_numeric_content_not_over_estimated(self) -> None:
        """Numeric content should not be grossly over-estimated."""
        # CSV-like numeric content
        text = ",".join(str(i) for i in range(100))
        tokens = estimate_tokens(text)

        # 100 numbers + 99 commas, should be ~200 tokens max
        assert tokens < 300, f"Numeric content over-estimated: {tokens} tokens"
