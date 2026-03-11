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
Token estimation utilities for LLM context management.

Provides fast, dependency-free heuristic token counting that works offline
without requiring tiktoken or network access. Optimized for CSV/tabular data
with ~8% average error compared to tiktoken on real-world datasets.

Example usage:
    >>> from datarobot_genai.core.utils.token_tracking import estimate_tokens
    >>> tokens = estimate_tokens("Hello, world!")
    >>> print(f"Estimated tokens: {tokens}")

    >>> # For DataFrames:
    >>> from datarobot_genai.core.utils.token_tracking import estimate_csv_rows_for_token_limit
    >>> csv_text, token_count = estimate_csv_rows_for_token_limit(df, max_tokens=5000)

    >>> # For LLM usage tracking:
    >>> from datarobot_genai.core.utils.token_tracking import TokenUsageTracker
    >>> tracker = TokenUsageTracker(strategy=HeuristicTokenCountingStrategy())
    >>> tracker.track_call(messages, response, model="gpt-4")
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any
from typing import Protocol
from typing import runtime_checkable

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for performance (module-level, compiled once)
# Uses [^\W\d_]+ to match Unicode letters (word chars minus digits and underscore)
# This correctly handles non-ASCII alphabets (Cyrillic, Arabic, CJK, etc.)
_TOKEN_PATTERN = re.compile(r"[^\W\d_]+|\d+|[^\w\s]")
_NEWLINE_PATTERN = re.compile(r"[\n\r\t]")

# Threshold for switching to sampling-based estimation (10 MB)
_SAMPLING_THRESHOLD = 10 * 1024 * 1024
_SAMPLE_SIZE = 100_000  # Size of each sample chunk
_SAMPLE_COUNT = 10  # Number of samples to take

# CJK Unicode ranges for special tokenization handling
# These scripts don't use spaces between words, so each character ≈ 1-2 tokens
_CJK_RANGES = (
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs (Chinese, Japanese Kanji, Korean Hanja)
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x3040, 0x309F),  # Hiragana (Japanese)
    (0x30A0, 0x30FF),  # Katakana (Japanese)
    (0xAC00, 0xD7AF),  # Hangul Syllables (Korean)
    (0x1100, 0x11FF),  # Hangul Jamo (Korean)
)


def _is_cjk_char(char: str) -> bool:
    """Check if a character is in CJK Unicode ranges."""
    code = ord(char)
    return any(start <= code <= end for start, end in _CJK_RANGES)


def _count_tokens_in_text(text: str) -> int:
    """Count tokens in a text string using regex-based tokenization.

    Internal helper that processes all tokens in a single pass.
    Handles both ASCII and Unicode text, with special handling for CJK scripts.
    """
    total = 0

    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group()
        first_char = token[0]

        if first_char.isdigit():
            # Numbers: ~3 digits per token (GPT tokenizers group digits)
            # e.g., "123" → 1 token, "123456" → 2 tokens
            total += max(1, (len(token) + 2) // 3)
        elif first_char.isalpha():
            # Check if this is CJK text (no word boundaries)
            if _is_cjk_char(first_char):
                # CJK: ~0.5-0.7 tokens per character on average (based on tiktoken analysis)
                # Using 0.85 as a safe multiplier that covers edge cases (Korean can hit 1.0)
                # while being much more accurate than the overly conservative 1.5
                total += max(1, int(len(token) * 0.85))
            else:
                # Latin, Cyrillic, Arabic, etc.: word-based tokenization
                word_len = len(token)
                if word_len <= 6:
                    # Common short words: typically 1 token
                    total += 1
                elif word_len <= 10:
                    # Medium words: often split into 2 subwords
                    total += 2
                else:
                    # Long words: roughly 1 token per 4 characters
                    total += (word_len + 3) // 4
        else:
            # Punctuation: each symbol is typically its own token
            total += 1

    # Newlines and tabs are typically separate tokens
    total += len(_NEWLINE_PATTERN.findall(text))

    # CSV optimization: BPE tokenizers merge common CSV patterns like ","
    # into single tokens. Count occurrences and subtract the savings.
    # Pattern '","' is 3 chars but typically 1 token (saves 2 tokens each)
    csv_pattern_count = text.count('","')
    if csv_pattern_count > 0:
        # Each '","' pattern: we counted " + , + " = 3 tokens
        # But tiktoken merges it to 1 token, so subtract 2 per occurrence
        total -= csv_pattern_count * 2

    return total


def estimate_tokens(text: str) -> int:
    """
    Estimate token count using a heuristic based on BPE tokenization patterns.

    This function provides a fast, dependency-free approximation of token counts
    without requiring tiktoken or network access.

    Algorithm based on empirical observations of GPT tokenizers:
    - Common words (≤6 chars): ~1 token each
    - Longer words: split into subword tokens (~4 chars per token)
    - Numbers: ~3 digits per token
    - Punctuation: usually separate tokens
    - Whitespace: merged with adjacent tokens (except newlines)
    - CJK text (Chinese/Japanese/Korean): ~0.85 tokens per character

    Unicode Support:
    - Handles all Unicode alphabets (Latin, Cyrillic, Arabic, Greek, etc.)
    - Special handling for CJK scripts which don't use word boundaries
    - Non-ASCII punctuation counted as individual tokens

    Performance:
    - Small texts (<10 MB): Full accurate counting, ~12 MB/sec
    - Large texts (≥10 MB): Sampling-based estimation, ~600+ MB/sec
      (processes 10 evenly-distributed 100KB samples and extrapolates)

    Accuracy (compared to tiktoken):
    - Real CSV data: ~8% average error (primary use case)
    - English prose: ±10-15%
    - Code/technical content: ±15-20%
    - Numeric-heavy content: ±5-10%
    - CJK text: ±15-25% (tends to overestimate, which is safer)

    Based on OpenAI's tokenizer guidelines:
    - ~4 characters per token for English text
    - ~0.75 tokens per word on average
    Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them

    For exact counts, use tiktoken or check API response usage data.

    Parameters
    ----------
    text : str
        Text to estimate tokens for

    Returns
    -------
    int
        Estimated number of tokens
    """
    if not text:
        return 0

    text_len = len(text)

    # For very large texts, use sampling for performance
    # This provides ~150x speedup with <0.01% accuracy loss
    if text_len > _SAMPLING_THRESHOLD:
        chunk_step = text_len // _SAMPLE_COUNT
        total_sample_tokens = 0
        total_sample_chars = 0

        for i in range(_SAMPLE_COUNT):
            start = i * chunk_step
            end = min(start + _SAMPLE_SIZE, text_len)
            chunk = text[start:end]

            total_sample_tokens += _count_tokens_in_text(chunk)
            total_sample_chars += len(chunk)

        # Extrapolate to full text
        if total_sample_chars > 0:
            tokens_per_char = total_sample_tokens / total_sample_chars
            return max(1, int(text_len * tokens_per_char))

    # For smaller texts, count all tokens accurately
    return max(1, _count_tokens_in_text(text))


def estimate_tokens_from_file(
    file_path: str,
    encoding: str = "utf-8",
    sample_size: int = _SAMPLE_SIZE,
    sample_count: int = _SAMPLE_COUNT,
) -> int:
    """
    Estimate token count from a file without loading it entirely into memory.

    This function is memory-efficient for large files (100MB+, even multi-GB).
    It reads only small samples from evenly-distributed positions in the file,
    then extrapolates to estimate total tokens.

    Memory usage: ~2-3MB regardless of file size (only samples are loaded).

    Parameters
    ----------
    file_path : str
        Path to the text file
    encoding : str, optional
        File encoding (default: utf-8)
    sample_size : int, optional
        Size of each sample chunk in bytes (default: 100KB)
    sample_count : int, optional
        Number of samples to take (default: 10)

    Returns
    -------
    int
        Estimated number of tokens

    Examples
    --------
    >>> tokens = estimate_tokens_from_file("large_dataset.csv")
    >>> print(f"Estimated {tokens:,} tokens")
    """
    file_size = os.path.getsize(file_path)

    if file_size == 0:
        return 0

    # For small files, read entirely (more accurate)
    if file_size <= _SAMPLING_THRESHOLD:
        with open(file_path, encoding=encoding) as f:
            return estimate_tokens(f.read())

    # For large files, sample from multiple positions
    chunk_step = file_size // sample_count
    total_sample_tokens = 0
    total_sample_chars = 0

    with open(file_path, encoding=encoding) as f:
        for i in range(sample_count):
            # Seek to position (approximate, may land mid-character for UTF-8)
            seek_pos = i * chunk_step
            f.seek(seek_pos)

            # Skip partial line/character at seek position
            if seek_pos > 0:
                f.readline()  # Discard partial line

            # Read sample chunk
            chunk = f.read(sample_size)
            if not chunk:
                continue

            total_sample_tokens += _count_tokens_in_text(chunk)
            total_sample_chars += len(chunk)

    # Extrapolate to full file
    if total_sample_chars > 0:
        # Estimate total characters from file size (approximate for UTF-8)
        # For UTF-8, average ~1.1 bytes per character for mixed content
        avg_bytes_per_char = (
            file_size / max(1, total_sample_chars) * (sample_size * sample_count / file_size)
        )
        estimated_total_chars = file_size / max(1.0, avg_bytes_per_char)

        tokens_per_char = total_sample_tokens / total_sample_chars
        return max(1, int(estimated_total_chars * tokens_per_char))

    return 1


def estimate_tokens_streaming(
    text_iterator: Any,  # Iterator[str] or Iterable[str]
    sample_every_n: int = 100,
    max_samples: int = 1000,
) -> tuple[int, int]:
    """
    Estimate tokens from a streaming text source (e.g., file lines, DataFrame rows).

    Memory-efficient for processing large datasets row-by-row without loading
    everything into memory. Samples every Nth item and extrapolates.

    Parameters
    ----------
    text_iterator : Iterable[str]
        Iterator yielding text strings (e.g., file lines, df rows)
    sample_every_n : int, optional
        Sample every Nth item (default: 100)
    max_samples : int, optional
        Maximum number of samples to collect (default: 1000)

    Returns
    -------
    tuple[int, int]
        (estimated_total_tokens, items_processed)

    Examples
    --------
    >>> with open("large_file.csv") as f:
    ...     tokens, lines = estimate_tokens_streaming(f)
    >>> print(f"Estimated {tokens:,} tokens in {lines:,} lines")

    >>> # For DataFrames:
    >>> df_iter = (row.to_csv(index=False, header=False) for _, row in df.iterrows())
    >>> tokens, rows = estimate_tokens_streaming(df_iter)
    """
    total_items = 0
    sampled_items = 0
    sampled_tokens = 0

    for i, text in enumerate(text_iterator):
        total_items += 1

        # Sample every Nth item
        if i % sample_every_n == 0 and sampled_items < max_samples:
            sampled_tokens += _count_tokens_in_text(str(text))
            sampled_items += 1

    if sampled_items == 0:
        return 0, total_items

    # Extrapolate
    avg_tokens_per_item = sampled_tokens / sampled_items
    estimated_total = int(avg_tokens_per_item * total_items)

    return estimated_total, total_items


def estimate_csv_rows_for_token_limit(
    df: Any,  # pandas.DataFrame
    max_tokens: int,
    initial_rows: int = 750,
) -> tuple[str, int]:
    """
    Estimate the optimal number of rows for CSV data to fit within token limit.

    Converts a DataFrame to CSV and iteratively reduces rows until the token
    count fits within the specified limit. Uses heuristic token estimation
    for fast, offline operation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to convert to CSV
    max_tokens : int
        Maximum allowed tokens for the CSV data
    initial_rows : int, optional
        Initial number of rows to try (default: 750)

    Returns
    -------
    tuple[str, int]
        (csv_string, final_token_count)

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"a": range(1000), "b": range(1000)})
    >>> csv_text, tokens = estimate_csv_rows_for_token_limit(df, max_tokens=5000)
    >>> print(f"CSV has {tokens} tokens")
    """
    df_csv = df.head(initial_rows).to_csv(index=False, quoting=1)
    csv_token_count = estimate_tokens(df_csv)

    if csv_token_count <= max_tokens:
        return df_csv, csv_token_count

    logger.warning(
        f"CSV data has {csv_token_count} tokens, exceeds limit of {max_tokens}. Reducing rows."
    )

    ratio = max_tokens / csv_token_count
    estimated_rows = int(initial_rows * ratio * 0.9)
    estimated_rows = max(100, estimated_rows)

    df_csv = df.head(estimated_rows).to_csv(index=False, quoting=1)
    final_token_count = estimate_tokens(df_csv)

    if final_token_count > max_tokens:
        estimated_rows = int(estimated_rows * 0.8)
        df_csv = df.head(estimated_rows).to_csv(index=False, quoting=1)
        final_token_count = estimate_tokens(df_csv)

    logger.info(
        f"Reduced CSV to {estimated_rows} rows ({final_token_count} tokens) "
        f"to fit within context window."
    )
    return df_csv, final_token_count


@runtime_checkable
class TokenCountingStrategy(Protocol):
    """Protocol for token counting strategies."""

    def count_tokens(
        self,
        messages: list[Any],  # list[ChatCompletionMessageParam]
        response: Any,
        model: str,
    ) -> tuple[int, int]:
        """
        Count prompt and completion tokens.

        Parameters
        ----------
        messages : list
            Input messages sent to LLM
        response : Any
            Response from LLM
        model : str
            Model name

        Returns
        -------
        tuple[int, int]
            (prompt_tokens, completion_tokens)
        """
        ...


class HeuristicTokenCountingStrategy:
    """
    Token counting using smart heuristic estimation.

    Uses a BPE-aware algorithm that considers word length, punctuation,
    numbers, and special characters. More accurate than simple char/4.
    No external dependencies required.

    Examples
    --------
    >>> strategy = HeuristicTokenCountingStrategy()
    >>> messages = [{"role": "user", "content": "Hello!"}]
    >>> prompt_tokens, completion_tokens = strategy.count_tokens(messages, response, "gpt-4")
    """

    def _count_text(self, text: str) -> int:
        """Count tokens in text."""
        return estimate_tokens(text)

    def _count_messages(self, messages: list[Any]) -> int:
        """Count tokens in messages."""
        total_tokens = 0
        for msg in messages:
            # Extract content based on message type
            role: str = ""
            content: str = ""

            if hasattr(msg, "get"):  # Dict-like
                role = str(msg.get("role", ""))
                content = str(msg.get("content", ""))
            else:
                # TypedDict attributes
                role = str(getattr(msg, "role", ""))
                content = str(getattr(msg, "content", ""))

            total_tokens += self._count_text(role)
            total_tokens += self._count_text(content)
            total_tokens += 4  # Message structure overhead

        return total_tokens

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from various response formats."""
        if hasattr(response, "content") and response.content:
            return str(response.content)
        if hasattr(response, "model_dump_json"):
            return str(response.model_dump_json())
        if hasattr(response, "__dict__"):
            return str(response.__dict__)
        return str(response)

    def count_tokens(
        self,
        messages: list[Any],  # list[ChatCompletionMessageParam]
        response: Any,
        model: str,  # noqa: ARG002 - kept for Protocol compatibility
    ) -> tuple[int, int]:
        """Count tokens using heuristic estimation."""
        prompt_tokens = self._count_messages(messages)
        response_text = self._extract_response_text(response)
        completion_tokens = self._count_text(response_text)

        return prompt_tokens, completion_tokens


# Backward compatibility alias - deprecated, use HeuristicTokenCountingStrategy
TiktokenCountingStrategy = HeuristicTokenCountingStrategy


class ApiResponseCountingStrategy:
    """
    Token counting from API response (preferred when available).

    Falls back to heuristic estimation if API response doesn't include usage data.

    Examples
    --------
    >>> strategy = ApiResponseCountingStrategy()
    >>> prompt_tokens, completion_tokens = strategy.count_tokens(messages, response, "gpt-4")
    """

    def __init__(self, fallback_strategy: TokenCountingStrategy | None = None) -> None:
        """
        Initialize with optional fallback strategy.

        Parameters
        ----------
        fallback_strategy : TokenCountingStrategy, optional
            Strategy to use if API response doesn't have usage data.
            Defaults to HeuristicTokenCountingStrategy.
        """
        if fallback_strategy is None:
            fallback_strategy = HeuristicTokenCountingStrategy()
        self.fallback_strategy: TokenCountingStrategy = fallback_strategy

    def count_tokens(
        self,
        messages: list[Any],  # list[ChatCompletionMessageParam]
        response: Any,
        model: str,
    ) -> tuple[int, int]:
        """Extract token counts from API response."""
        # Try to get usage from response
        usage = self._extract_usage(response)

        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)

            if prompt_tokens > 0 and completion_tokens > 0:
                logger.debug(
                    f"Using API response token counts: {prompt_tokens} prompt, "
                    f"{completion_tokens} completion"
                )
                return prompt_tokens, completion_tokens

        # Fallback to heuristic estimation
        logger.debug("API usage data not available, using heuristic estimation")
        return self.fallback_strategy.count_tokens(messages, response, model)

    @staticmethod
    def _extract_usage(response: Any) -> Any | None:
        """Extract usage data from various response formats."""
        # Try instructor response format
        if hasattr(response, "_raw_response"):
            raw = response._raw_response
            if hasattr(raw, "usage"):
                return raw.usage

        # Try direct usage attribute
        if hasattr(response, "usage"):
            return response.usage

        # Try dict format
        if isinstance(response, dict) and "usage" in response:
            return response["usage"]

        return None


class TokenUsageTracker:
    """
    Accumulates token usage across multiple LLM calls.

    Useful for tracking total token consumption in a session or workflow.

    Examples
    --------
    >>> tracker = TokenUsageTracker(strategy=HeuristicTokenCountingStrategy())
    >>> tracker.track_call(messages, response, "gpt-4")
    >>> print(tracker.to_dict())
    """

    def __init__(self, strategy: TokenCountingStrategy) -> None:
        """
        Initialize tracker with counting strategy.

        Parameters
        ----------
        strategy : TokenCountingStrategy
            Token counting strategy to use for tracking.
        """
        self.strategy = strategy
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self.model = ""

    def track_call(
        self,
        messages: list[Any],  # list[ChatCompletionMessageParam]
        response: Any,
        model: str,
    ) -> None:
        """
        Track token usage from an LLM call.

        Parameters
        ----------
        messages : list
            Input messages
        response : Any
            LLM response
        model : str
            Model name
        """
        prompt_tokens, completion_tokens = self.strategy.count_tokens(messages, response, model)

        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.call_count += 1
        if model:
            self.model = model

        logger.debug(
            f"Token tracker: +{prompt_tokens} prompt, +{completion_tokens} completion "
            f"(total calls: {self.call_count}, total tokens: {self.total_tokens})"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary.

        Returns
        -------
        dict
            Dictionary with prompt_tokens, completion_tokens, total_tokens,
            call_count, and model.
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "model": self.model,
        }


def count_messages_tokens(
    messages: list[Any],  # list[ChatCompletionMessageParam]
    model: str = "",  # noqa: ARG001 - kept for API compatibility
) -> int:
    """
    Count tokens in a list of chat messages using heuristic estimation.

    Parameters
    ----------
    messages : list
        List of chat messages (OpenAI format)
    model : str, optional
        Model name (unused, kept for API compatibility)

    Returns
    -------
    int
        Estimated token count for all messages
    """
    strategy = HeuristicTokenCountingStrategy()
    return strategy._count_messages(messages)
