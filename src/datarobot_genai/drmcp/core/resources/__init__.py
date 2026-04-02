"""Standard MCP resources for DataRobot entities.

Importing this package registers dataset://, deployment://, and model://
resource handlers on the global MCP instance. Any MCP client can then
discover and read these resources without panel-specific knowledge.

Resources registered:
  dataset://            — list all accessible datasets
  dataset://{id}        — metadata + sample rows for a dataset
  deployment://         — list all deployments
  deployment://{id}     — deployment info (model, target, features)
  model://              — list all registered models
  model://{id}          — model details and metrics
"""
from . import datasets, deployments, models  # noqa: F401 — triggers @mcp.resource registration
