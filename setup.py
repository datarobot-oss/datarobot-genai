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
into all other extras except 'drmcp' at build time.
"""

from setuptools import setup

# Core dependencies shared across extras. These are merged into other extras.
core = [
    "requests>=2.32.4,<3.0.0",
    "datarobot>=3.10.0,<4.0.0",
    "datarobot-predict>=1.13.2,<2.0.0",
    "openai>=1.76.2,<2.0.0",
    "ragas>=0.3.8,<0.4.0",
    "pyjwt>=2.10.1,<3.0.0",
    "pypdf>=6.1.3,<7.0.0",  # CVE BUZZOK-28182
    "opentelemetry-instrumentation-requests>=0.43b0,<1.0.0",
    "opentelemetry-instrumentation-aiohttp-client>=0.43b0,<1.0.0",
    "opentelemetry-instrumentation-httpx>=0.43b0,<1.0.0",
    "opentelemetry-instrumentation-openai>=0.40.5,<1.0.0",
    "opentelemetry-instrumentation-threading>=0.43b0,<1.0.0",
    "ag-ui-protocol>=0.1.9,<0.2.0",
    "pyarrow==21.0.0",
]

dragent = core + [
    "nvidia-nat==1.4.1; python_version >= '3.11'",
    "nvidia-nat-opentelemetry==1.4.1; python_version >= '3.11'",
]

crewai = core + [
    "anthropic~=0.71.0,<1.0.0",  # Needed for integration with anthropic endpoints
    "azure-ai-inference>=1.0.0b9,<2.0.0",  # Needed for integration with azure endpoints
    "crewai[litellm]>=1.1.0,<2.0.0",
    "crewai-tools[mcp]>=0.69.0,<0.77.0",
    "opentelemetry-instrumentation-crewai>=0.40.5,<1.0.0",
    "pybase64>=1.4.2,<2.0.0",
]

langgraph = core + [
    "langchain-mcp-adapters>=0.1.12,<0.2.0",
    "langgraph>=1.0.0,<1.1.0",
    "langgraph-prebuilt>=1.0.0,<1.1.0",
    "opentelemetry-instrumentation-langchain>=0.40.5,<1.0.0",
]

llamaindex = core + [
    "llama-index>=0.14.0,<0.15.0",
    "llama-index-core>=0.14.0,<0.15.0",
    "llama-index-llms-langchain>=0.6.1,<0.8.0",
    "llama-index-llms-litellm>=0.4.1,<0.7.0",  # Sync nat dependency if possible too
    "llama-index-llms-openai>=0.6.0,<0.7.0",
    "llama-index-tools-mcp>=0.1.0,<0.5.0",
    "opentelemetry-instrumentation-llamaindex>=0.40.5,<1.0.0",
    "pypdf>=6.0.0,<7.0.0",
]

nat = core + [
    "nvidia-nat==1.4.1; python_version >= '3.11'",
    "nvidia-nat-a2a==1.4.1; python_version >= '3.11'",
    "nvidia-nat-opentelemetry==1.4.1; python_version >= '3.11'",
    "nvidia-nat-langchain==1.4.1; python_version >= '3.11'",
    "nvidia-nat-llama-index==1.4.1; python_version >= '3.11'",
    "nvidia-nat-mcp==1.4.1; python_version >= '3.11'",
    "crewai>=1.1.0; python_version >= '3.11'",
    "llama-index-llms-litellm>=0.4.1,<0.7.0",  # Need this to support datarobot-llm plugin
    "opentelemetry-instrumentation-crewai>=0.40.5,<1.0.0",
    "opentelemetry-instrumentation-llamaindex>=0.40.5,<1.0.0",
    "opentelemetry-instrumentation-langchain>=0.40.5,<1.0.0",
    "anyio==4.11.0",
]

pydanticai = core + [
    "pydantic-ai-slim[ag-ui,anthropic,bedrock,cli,cohere,evals,fastmcp,google,groq,huggingface,logfire,mcp,mistral,openai,retries,vertexai]>=1.0.5,<1.9.0",
]

# drmcp is standalone set of dependencies for MCP Server only
drmcp = core + [
    "datarobot-asgi-middleware>=0.2.0,<1.0.0",
    "python-dotenv>=1.1.0,<2.0.0",
    "boto3>=1.34.0,<2.0.0",
    "httpx>=0.28.1,<1.0.0",
    "tavily-python>=0.7.20,<1.0.0",
    "pandas>=2.2.3,<3.0.0",
    "perplexityai>=0.27,<1.0",
    "pydantic>=2.6.1,<3.0.0",
    "pydantic-settings>=2.1.0,<3.0.0",
    "opentelemetry-api>=1.22.0,<2.0.0",
    "opentelemetry-sdk>=1.22.0,<2.0.0",
    "opentelemetry-exporter-otlp>=1.22.0,<2.0.0",
    "opentelemetry-exporter-otlp-proto-http>=1.22.0,<2.0.0",
    "aiohttp>=3.9.0,<4.0.0",
    "aiohttp-retry>=2.8.3,<3.0.0",
    "aiosignal>=1.3.1,<2.0.0",
    "fastmcp>=2.13.0.2,<3.0.0",
    "beautifulsoup4>=4.14.3,<5.0.0",
]

# auth is standalone set of dependencies for auth utilities only
auth = [
  "datarobot[auth]>=3.10.0,<4.0.0",
  "aiohttp>=3.9.0,<4.0.0",
  "pydantic>=2.6.1,<3.0.0",
]


extras_require = {
    "core": core,
    "crewai": crewai,
    "langgraph": langgraph,
    "llamaindex": llamaindex,
    "nat": nat,
    "pydanticai": pydanticai,
    "auth": auth,
    "drmcp": drmcp,
    "dragent": dragent,
}

setup(extras_require=extras_require)
