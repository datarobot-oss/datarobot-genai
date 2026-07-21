# Copyright 2025 DataRobot, Inc. and its affiliates.
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

"""Setup script defining optional dependencies (extras) for the package.

This script defines all extras and automatically merges 'core' dependencies
into all other extras except standalone extras (`auth`, `drtools`, `drmcpbase`, `drmcp`, `drmcputils`, `eval`)
at build time.
"""

from setuptools import setup

# Core dependencies shared across extras. These are merged into other extras except standalone extras.
core = [
    "requests>=2.32.4,<3.0.0",
    "datarobot>=3.17.0,<4.0.0",
    "datarobot-predict>=1.13.2,<2.0.0",
    "openai>=2.0.0,<3.0.0",
    "ragas>=0.4.3,<0.5.0",
    "pyjwt>=2.12.0,<3.0.0",  # CVE-2026-32597 fixed in 2.12.0
    "opentelemetry-instrumentation-requests>=0.64b0,<1.0.0",
    "opentelemetry-instrumentation-aiohttp-client>=0.64b0,<1.0.0",
    "opentelemetry-instrumentation-httpx>=0.64b0,<1.0.0",
    "opentelemetry-instrumentation-openai>=0.62.1,<1.0.0",
    "opentelemetry-instrumentation-threading>=0.64b0,<1.0.0",
    "datarobot-moderations[all]>=11.2.33,<12.0.0",
    # Keep this version in sync with all consumers of agent messages e.g. the fastapi_server of the
    # agent application template
    "ag-ui-protocol==0.1.15",
    "pyarrow>=23.0.1,<24.0.0",  # CVE-2026-25087 fixed in 23.0.1
    "colorama>=0.4.6,<1.0.0",
    "httpx-retries>=0.4.0",
]

crewai = core + [
    "anthropic~=0.71.0,<1.0.0",  # Needed for integration with anthropic endpoints
    "azure-ai-inference>=1.0.0b9,<2.0.0",  # Needed for integration with azure endpoints
    "crewai[litellm]>=1.11.0",
    "litellm>=1.91.1,<2.0.0",
    "crewai-tools[mcp]>=0.69.0,<0.77.0",
    "mcpadapt>=0.1.9",  # imported directly by crewai/mcp.py
    "nvidia-nat-crewai==1.7.0",
    "opentelemetry-instrumentation-crewai>=0.62.1,<1.0.0",
    "pybase64>=1.4.2,<2.0.0",
]

langgraph = core + [
    "langchain-mcp-adapters>=0.1.12,<0.2.0",
    "langgraph>=1.0.0,<2.0.0",
    "langgraph-prebuilt>=1.0.0,<2.0.0",
    "litellm>=1.91.1,<2.0.0",
    "nvidia-nat-langchain==1.7.0",
    "opentelemetry-instrumentation-langchain>=0.62.1,<1.0.0",
]

llamaindex = core + [
    "llama-index>=0.14.0,<0.15.0",
    "llama-index-core>=0.14.0,<0.15.0",
    "llama-index-llms-langchain>=0.8.0,<1.0.0",
    "llama-index-llms-litellm>=0.4.1,<0.7.0",  # Sync nat dependency if possible too
    "litellm>=1.91.1,<2.0.0",
    "llama-index-llms-openai>=0.6.0,<0.7.0",
    "llama-index-tools-mcp>=0.1.0,<0.5.0",
    "nvidia-nat-llama-index==1.7.0",
    "opentelemetry-instrumentation-llamaindex>=0.62.1,<1.0.0",
    "pypdf>=6.10.1,<7.0.0",  # CVE-2026-40260 fixed in 6.10.0; GHSA-jj6c-8h6c-hppx in 6.10.1
]

dragent = core + [
    "nvidia-nat==1.7.0",
    "nvidia-nat-a2a==1.7.0",
    "nvidia-nat-opentelemetry==1.7.0",
    "nvidia-nat-langchain==1.7.0",  # NAT built-in agents require this
    "nvidia-nat-mcp==1.7.0",
    "anyio==4.11.0",
    "mem0ai>=1.0.4,<2.0.0",
    "starlette>=1.0.1",  # CVE fix
]

# auth is standalone set of dependencies for auth utilities only
auth = [
  "datarobot[auth]>=3.17.0,<4.0.0",
  "aiohttp>=3.13.3,<4.0.0",  # CVE-2025-69229 & CVE-2025-69230 fixed in 3.13.3
  "pydantic>=2.6.1,<3.0.0",
  "httpx>=0.28.1,<1.0.0",
  "pyjwt[crypto]>=2.12.0,<3.0.0",
  "okta-client-python>=0.2.0,<1.0.0",
]

# drmcputils is a leaf subpackage: no imports from other datarobot_genai subpackages.
drmcputils = auth + [
    "datarobot[fs]>=3.17.0,<4.0.0",
]

# drtools: no subpackages dependencies other than auth and drmcputils.
# polars for internal tabular data; pandas only at predict API boundary (datarobot-predict).
drtools =  drmcputils + [
    "beautifulsoup4>=4.12.0,<5.0.0",
    "httpx>=0.28.1,<1.0.0",
    "tavily-python>=0.7.20,<1.0.0",
    "perplexityai>=0.27,<1.0",
    "pypdf>=6.10.1,<7.0.0",  # CVE-2026-40260 fixed in 6.10.0; GHSA-jj6c-8h6c-hppx in 6.10.1
    "polars>=1.0.0,<2.0.0",
    # Required indirectly by polars->pandas conversion.
    "pyarrow>=23.0.1,<24.0.0",  # CVE-2026-25087 fixed in 23.0.1
    "python-dateutil>=2.9.0,<3.0.0",
    "datarobot-predict>=1.13.2,<2.0.0",
    "pydantic>=2.6.1,<3.0.0",
    "aiohttp>=3.13.3,<4.0.0",  # CVE-2025-69229 & CVE-2025-69230 fixed in 3.13.3
    # OTel API/SDK + OTLP/HTTP exporter: sandbox SLI metrics (drtools observability,
    # drmcpbase metrics bootstrap) import these at module level.
    "opentelemetry-api>=1.22.0,<2.0.0",
    "opentelemetry-sdk>=1.22.0,<2.0.0",
    "opentelemetry-exporter-otlp-proto-http>=1.22.0,<2.0.0",
]

# eval is standalone set of dependencies for evaluation utilities only (no core).
eval_deps = [
    "nemo-evaluator-launcher",
    "litellm>=1.91.1,<2.0.0",
    "pyyaml>=6.0",
]

# drmcpbase is standalone set of dependencies for MCP Servers only (no core).
drmcpbase = drmcputils + [
    "starlette>=1.0.1",  # CVE-2026-48710 fixed in 1.0.1
    "fastmcp>=3.4.1,<4.0.0",
    "aiohttp>=3.13.3,<4.0.0",
    "aiohttp-retry>=2.8.3,<3.0.0",
    "cachetools>=5.0.0,<8.0.0",
    # OTel API/SDK + OTLP/HTTP exporter: sandbox SLI metrics (drtools observability,
    # drmcpbase metrics bootstrap) import these at module level.
    "opentelemetry-api>=1.22.0,<2.0.0",
    "opentelemetry-sdk>=1.22.0,<2.0.0",
    "opentelemetry-exporter-otlp-proto-http>=1.22.0,<2.0.0",
]

# drmcp is standalone set of dependencies for MCP Template Server only (no core), only depends on drmcpbase and drtools.
drmcp = drmcpbase + drtools + [
    "requests>=2.32.4,<3.0.0",
    "openai>=2.0.0,<3.0.0",
    "pyjwt>=2.12.0,<3.0.0",
    "opentelemetry-instrumentation-requests>=0.64b0,<1.0.0",
    "opentelemetry-instrumentation-aiohttp-client>=0.64b0,<1.0.0",
    "opentelemetry-instrumentation-httpx>=0.64b0,<1.0.0",
    "rich>=13.0.0,<16.0.0",
    "datarobot-asgi-middleware>=0.2.0,<1.0.0",  # not imported in drmcp; used when running server in DataRobot ASGI env
    "python-dotenv>=1.1.0,<2.0.0",
    "pydantic-settings>=2.1.0,<3.0.0",
    "opentelemetry-api>=1.43.0,<2.0.0",
    "opentelemetry-sdk>=1.43.0,<2.0.0",
    "opentelemetry-exporter-otlp>=1.43.0,<2.0.0",
    "opentelemetry-exporter-otlp-proto-http>=1.43.0,<2.0.0",
    "async-lru>=2.3.0",
]

extras_require = {
    "core": core,
    "crewai": crewai,
    "langgraph": langgraph,
    "llamaindex": llamaindex,
    "auth": auth,
    "eval": eval_deps,
    "drmcpbase": drmcpbase,
    "drmcputils": drmcputils,
    "drmcp": drmcp,
    "drtools": drtools,
    "dragent": dragent,
}

setup(extras_require=extras_require)
