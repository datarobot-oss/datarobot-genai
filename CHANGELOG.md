# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.7.2
- Add `temperature` parameter support to LLM MCP clients (`BaseLLMMCPClient`, `DRLLMGatewayMCPClient`): read from config dict and forwarded to `chat.completions.create`
- Add `LLM_TEMPERATURE` env var support in `get_openai_llm_client_config()` and `get_dr_llm_gateway_client_config()` to control LLM temperature in acceptance tests
- Switch tool parameter matching in `ToolBaseE2E` from exact equality to subset matching (`_check_dict_params_match`) to reduce test flakiness

## 0.7.1
- Migrate Data MCP tools from wren mcp (MODEL-22804)

## 0.7.0
- **Breaking**: drop Python 3.10 support because of using NAT in all agents
- Pass forwarded DataRobot headers to agents, MCP clients, and LLMs
- Pass authorization context to agents and MCP clients
- Register MCP function group per user as it depends on the current user authorization context
- Rework dependencies in order to install only necessary libraries

## 0.6.21
- Added `x-datarobot-authorization` to `HEADER_TOKEN_CANDIDATE_NAMES` to fix auth when connecting through the API gateway

## 0.6.20
- Fixed an issue where the API token loaded via an environment variable was not properly serialized in NAT

## 0.6.19
- Enable A2A endpoints for per-user workflows with configurable skills via `DRAgentA2AConfig`
- **Breaking**: `DRAgentFastApiFrontEndConfig.a2a` type changed from `A2AFrontEndConfig` to `DRAgentA2AConfig`; update `workflow.yaml` by nesting the existing A2A fields under `server:`
- Added new data tools: get_dataset_details, list_datastores, browse_datastore, query_datastore
- Added "daria" tag to existing overlapping tools

## 0.6.17
- Fixed CVE-2026-25580: removed unused `pydantic-ai-slim` dependency and `pydanticai` install extra
- Added e2e tests for dragent server covering streaming, tool use, and MCP integration
- Added CI workflow for e2e tests with path-based triggers across langgraph, crewai, llamaindex, and nat

## 0.6.16
- Align MCP OpenTelemetry spans with OTel semantic conventions.

## 0.6.15
- Added Agent2Agent (A2A) server endpoints to `DRAgentFastApiFrontEndPluginWorker`, mounted at `/a2a`.
- Extended DRAgentFastApiFrontEndConfig with configuration options for the A2A server.
- A2A endpoints can be enabled by the `expose_a2a_server_endpoints` setting in the workflow.yaml file.
- Added `per_user_tool_calling_agent` workflow type
- Fixed `ToolCallArgsEvent.delta` encoding

## 0.6.14
- Fix loading JSON schemas from the package directory in DRUM adapter to work from wheel or source
- Fix dynamic tool deployment registration to filter deployments with `tool` tag name and value using strict AND logic
- Fix configuration parsing to correctly disable predictive tools when `MCP_CLI_CONFIGS` is empty

## 0.6.13
- Added `x-datarobot-identity-token` to `HEADER_TOKEN_CANDIDATE_NAMES`

## 0.6.12
- Add Mem0 for storing agent memory capabilities

## 0.6.11
- Fixed AG-UI event serialization

## 0.6.10
- Updated cursorbot review instructions with correct names

## 0.6.9
- Added cursorbot review instructions

## 0.6.8
- Make model required for MCP clients acceptance tests

## 0.6.7
- When getting api key in perplexity/tavily try to get it from 2 different headers (as fallback)

## 0.6.6
- Refactor drtools to use Polars for data handling, removing pandas dependency
- Enhance acceptance tests for realtime predictions with inline CSV dataset support

## 0.6.5
- Update task file to enable integration tests

## 0.6.4
- Created prompt stubs for MCP integration tests

## 0.6.3
- Created model stubs for MCP integration tests

## 0.6.2
- Created MCP predict test stubs and added stub predict client attribute integrated with MCP server

## 0.6.1
- Add e2e test scaffolding for CrewAI, LangGraph, and LlamaIndex agent frameworks

## 0.6.0
- Unify `InvokeReturn` type from `str | Event` to `Event` across all agent implementations
- All agents (CrewAI, LangGraph, LlamaIndex, NAT) now emit AG-UI lifecycle events
- Remove `str` code path from streaming/completions layer

## 0.5.15
- Restructure the tools and move them to drtools instead of drmcp.tools

## 0.5.14
- Fix unit tests to not depend on .env
- Update MCP dependencies to remove core extra
- Fix root conftest loads without error for drmcp tests by using inline imports.

## 0.5.13
- Add `dragent`: frontserver for DataRobot Agents
- Add placeholder for E2E tests of `dragent` and `drmcp`

## 0.5.12
Update MCP item metadata related enums
- DataRobotMCPToolCategory
- DataRobotMCPPromptCategory
- DataRobotMCPResourceCategory

## 0.5.11
- Created DR MCP test stubs and added stub `client` attribute integrated with MCP server

## 0.5.10
- Add DataRobotMCPServer APIs to retrieve registered tools/prompts/resources.

## 0.5.9
- Add MCP tool `deploy_custom_model` for deploying custom inference models (e.g. `.pkl`, `.joblib`) to DataRobot MLOps
- Custom model deployment: validation for prediction servers (consistent with `deploy_model`), optional execution environment, model file discovery in folder.

## 0.5.8
- Added "DR docs" tools: a tool for searching DataRobot Agentic AI docs and returning most relevant doc pages (includes title, URL, content) using TF-IDF, and a tool for fetching any DataRobot docs page. Note: only supported for English documentation, not Japanese.

## 0.5.7

- Updated NAT MCP client for 1.4.1 changes
- Update default transport for NAT MCP client to `streamable_http`
- Log error and fall back to empty function group when NAT MCP client is misconfigured

## 0.5.6

- Bump NAT libraries to 1.4.1
- Add `nvidia-nat-a2a` as a dependency for the `nat` extra

## 0.5.5

## 0.5.5
- Added upper bound to `crewai` dependency (`>=1.1.0,<2.0.0`)

## 0.5.4

- Added dr_mcp_prompt as prompt function decorator
- Added dr_mcp_resource as resource function decorator

## 0.5.3

- Added chat history support for all agent types (CrewAI, LangGraph, LlamaIndex, NAT)
- History is opt-in per agent; configurable via `max_history_messages` constructor param or `DATAROBOT_GENAI_MAX_HISTORY_MESSAGES` env var (default: 20)

## 0.5.2

- Dependency groups are converted into the optional `extra` options for installation, without list of default dependencies.

## 0.5.1

- Added the CHANGELOG.md file
- GitHub Action to verify changelog entries in pull requests

## 0.5.0

- No dependencies installed by default
- Optional `extra` options were converted into dependency groups and require `uv` for installation
- Added `auth` dependency group for authentication utilities
- Pandas dependency moved from the `core` to the `drmcp` dependency group
