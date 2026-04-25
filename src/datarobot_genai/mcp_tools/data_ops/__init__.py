"""Generic data operation tools for filtering, aggregating, sorting, and transforming datasets.

Importing this package registers filter_data, aggregate_data, sort_data,
and transform_data in the mcp_tools registry. These are generic operations
that work on any DataRobot dataset — not panel-specific.
"""
from . import aggregate, filter, sort, transform  # noqa: F401 — triggers register_tool() calls
