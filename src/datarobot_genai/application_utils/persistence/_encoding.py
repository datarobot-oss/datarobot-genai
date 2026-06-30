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

"""Description codec for range-key encoding in the Memory Service ORM.

The Memory Service ``description`` field supports only **case-insensitive
substring** match (minimum 3 characters).  The ORM encodes ``DRRangeKey``
values into a delimiter scheme that makes substring search equivalent to an
**anchored prefix-of-hierarchy** match::

    description = "//" + esc(prefix) + "/" + esc(k1) + "/" + … + esc(kN) + "/"

Encoding rules
--------------
* ``esc(v)`` percent-encodes ``%`` first, then ``/``, so values cannot inject
  delimiters (single ``/`` is the segment separator; ``//`` is the start anchor).
* Leading ``//`` = start anchor.  Two consecutive slashes can appear **only** here
  (single ``/`` is the segment delimiter and ``esc()`` turns any ``/`` inside
  values to ``%2F``), so a search for ``//prefix/`` cannot match at an internal
  boundary.
* Trailing ``/`` after **every** segment (including the last) = tail boundary.
  This prevents partial-segment matches: ``foo/12`` will NOT match ``foo/123``.

With both anchors, a ``description`` substring search is equivalent to a
prefix-of-hierarchy query:

.. code-block:: text

    stored:  //chat/acme/billing/
    search:  //chat/acme/         → matches  (all sessions under acme)
    search:  //chat/ac/           → no match (segment boundary respected)
    search:  //chat/acme/billing/ → matches  (exact leaf match)

Known limitation
----------------
The Memory Service filter is **case-insensitive**, so range-key values
differing only in case collide (``Foo`` and ``foo`` are treated as the same
prefix).  This is a v1 documented limitation; no fix is planned.
"""

from __future__ import annotations

_MAX_DESCRIPTION_LEN: int = 1000
_MIN_QUERY_LEN: int = 3  # service minimum for description substring filter


def _esc(value: str) -> str:
    """Percent-encode ``%`` and ``/`` in a range-key or prefix value."""
    return value.replace("%", "%25").replace("/", "%2F")


def _unesc(value: str) -> str:
    """Decode a percent-encoded segment (exact reverse of ``_esc``)."""
    return value.replace("%2F", "/").replace("%25", "%")


def validate_range_key(key_name: str, value: str) -> None:
    """Raise ``ValueError`` when a range-key value would break the encoding.

    Parameters
    ----------
    key_name : str
        Field name, used in the error message.
    value : str
        The range-key value to validate.

    Raises
    ------
    ValueError
        If ``value`` is empty.
    """
    if not value:
        raise ValueError(
            f"Range key {key_name!r} must not be empty — an empty segment would "
            "produce a mid-string '//' and break the description start anchor."
        )


def build_description(prefix: str, values: list[str]) -> str:
    """Build the encoded ``description`` string from a prefix and range-key values.

    Works for both **storage** (all range keys provided) and **querying** (a
    leading prefix of range keys provided — the resulting substring is anchored
    at the start).

    Parameters
    ----------
    prefix : str
        ``DRSession.__description_prefix__``.
    values : list[str]
        Ordered range-key values.  May be empty (produces ``"//prefix/"``).

    Returns
    -------
    str
        Encoded description string.

    Raises
    ------
    ValueError
        If the encoded string exceeds 1000 characters.

    Examples
    --------
    >>> build_description("chat", ["acme", "billing"])
    '//chat/acme/billing/'
    >>> build_description("chat", ["acme"])
    '//chat/acme/'
    >>> build_description("chat", [])
    '//chat/'
    >>> build_description("acme/corp", ["key/val"])
    '//acme%2Fcorp/key%2Fval/'
    """
    parts = [_esc(prefix)] + [_esc(v) for v in values]
    encoded = "//" + "/".join(parts) + "/"
    if len(encoded) > _MAX_DESCRIPTION_LEN:
        raise ValueError(
            f"Encoded description ({len(encoded)} chars) exceeds the "
            f"{_MAX_DESCRIPTION_LEN}-character limit. Shorten the prefix or "
            "range-key values."
        )
    return encoded


def build_query_description(prefix: str, values: list[str]) -> str:
    """Build the description substring used as the ``description`` filter in a list query.

    Identical to ``build_description`` but also enforces the service minimum
    of 3 characters for the ``description`` query parameter.

    Parameters
    ----------
    prefix : str
        ``DRSession.__description_prefix__``.
    values : list[str]
        Leading range-key values to filter on (may be empty).

    Returns
    -------
    str
        Encoded description substring (≥ 3 characters).

    Raises
    ------
    ValueError
        If the result is shorter than 3 characters.
    """
    encoded = build_description(prefix, values)
    if len(encoded) < _MIN_QUERY_LEN:
        raise ValueError(
            f"Encoded description query {encoded!r} is {len(encoded)} characters, "
            f"below the {_MIN_QUERY_LEN}-character minimum required by the Memory Service "
            "description filter.  Use a longer prefix."
        )
    return encoded


def parse_description(prefix: str, description: str, n: int) -> list[str]:
    """Extract ``n`` range-key values from a stored description.

    Parameters
    ----------
    prefix : str
        ``DRSession.__description_prefix__``.
    description : str
        Raw ``description`` value from the wire response.
    n : int
        Number of range-key segments to extract.

    Returns
    -------
    list[str]
        Decoded range-key values (length ``n``).

    Raises
    ------
    ValueError
        If ``description`` was not produced by ``build_description`` with the
        given ``prefix``, or if fewer than ``n`` segments are present.

    Examples
    --------
    >>> parse_description("chat", "//chat/acme/billing/", 2)
    ['acme', 'billing']
    >>> parse_description("chat", "//chat/acme/", 1)
    ['acme']
    """
    expected_start = "//" + _esc(prefix) + "/"
    if not description.startswith(expected_start):
        raise ValueError(
            f"Description {description!r} does not start with the expected anchor "
            f"{expected_start!r}.  It was not produced by this ORM with prefix "
            f"{prefix!r}."
        )
    rest = description[len(expected_start) :]
    # Strip the trailing slash that follows the last segment
    if rest.endswith("/"):
        rest = rest[:-1]
    segments = rest.split("/") if rest else []
    if len(segments) < n:
        raise ValueError(
            f"Expected {n} range-key segments in {description!r} for prefix "
            f"{prefix!r}, but found only {len(segments)}."
        )
    return [_unesc(s) for s in segments[:n]]
