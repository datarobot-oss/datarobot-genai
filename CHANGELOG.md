# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.14.7
- Crew AI parameters

## 0.14.6
- Fixed MCP tool calls to deployments silently swallowing HTTP error responses instead of surfacing them as tool errors.

## 0.14.5
- Relaxed `langgraph` and `langgraph-prebuilt` version constraints from `<1.1.0` to `<2.0.0`

## 0.14.4
- Pinned crewai to 1.11.0: fixed is_litellm issues and CrewAI attempts to load a native provider

## 0.14.3
- Added `nat dragent` shell frontend for NAT - provides `nat dragent serve`, `nat dragent run`, and `nat dragent query` commands
- MCP tool loading now gracefully falls back to empty tools on connection errors (CrewAI, LangGraph, LlamaIndex)

## 0.14.2
- Implemented functions to instantiate LLM classes for LLM Gateway, deployment, NIM, and external

## 0.14.1
- Added opt-in memory retrieval and storage to the LlamaIndex base agent when input messages declare `{memory}`.

## 0.14.0
- Reworked agents to only accept `llm`: its not the job of an agent to instantiate the LLM
- Implemented functions to convert native agentic primitives into DataRobot agents

## 0.13.3
- Enabled functionalities of synchronizing MCP item metadata within MCP server (still behind the feature flag).

## 0.13.2
- Updated to remove prints, but instead use `logging` instead in our agents outputs

## 0.13.1
- Added opt-in memory retrieval and storage to the CrewAI base agent when kickoff inputs declare `memory`.

## 0.13.0
- Upgraded fastmcp from 2.x to 3.2.0+ (CVE-2026-32871, CVE-2026-27124, CVE-2025-64340)
- Replaced `on_duplicate_tools`/`on_duplicate_prompts` with unified `on_duplicate` (fastmcp 3.x API)
- Replaced `enabled` parameter with `mcp.disable()` for tool/prompt/resource registration (fastmcp 3.x API)
- Made `Context.set_state`/`get_state` calls async (fastmcp 3.x API)

## 0.12.0
- **Breaking** Update Langgraph template to support mem0 store and retrieve long term memory

## 0.11.0
- Reworked legacy interface to support mcp_tool_context being external to an agent.

## 0.10.3
- Updated `openai` and `litellm`, and removed unused transitive dependencies.

## 0.10.2
- Fixed multi-turn tool-calling by preserving assistant `tool_calls` on `AssistantMessage` and retaining tool-call metadata in chat history summaries.

## 0.10.1
- Fixed multi-turn tool-calling: preserving tool_calls on AssistantMessage and passing structured native messages to all agent frameworks when conversation history is present.

## 0.10.0
- **Breaking**: Decoupled agents and MCP tools. Now mcp_tools_context has to be called explicitly outside of an agent invoke.
- Reworked `mcp_tools_context`arguments to use `MCPConfig` explicitly.
- Fixed issue with `RuntimeError: generator didn't stop after athrow()` in MCP context.
- Fix CrewAI streaming steps and reasoning: enable CrewAI streaming by default
- Unify logging for CrewAI

## 0.9.2
- Added AG-UI Events for CrewAI

## 0.9.1
- Suppressed known-harmless NAT warnings

## 0.9.0
- Removed side-effects in MCPConfig
- Renamed datarobot_genai.core.mcp.common to datarobot_genai.core.mcp.config

## 0.8.17
- Fixed prompt for calculator to make output for LlamaIndex agents stable

## 0.8.16
- **Security**: Upgraded `aiohttp>=3.13.3` to fix CVE-2025-69229 (DoS via chunked messages) and CVE-2025-69230 (cookie parser warning storm)
- **Security**: Upgraded `pypdf>=6.9.2` to fix CVE-2026-33699 (infinite loop in DictionaryObject recovery) and CVE-2026-33123 (inefficient stream decoding)
- **Security**: Upgraded `pyjwt>=2.12.0` to fix CVE-2026-32597 in core and drmcp dependencies
- **Security**: Verified authlib CVE-2026-27962 is patched via fastmcp transitive dependency (1.6.9+)

## 0.8.15
- Did a major refactor to decouple `drtools` from `drmcp`
- Added dependency lint check task to the ci
- Moved auth/token extraction (`_extract_token_from_headers`, `_extract_token_from_auth_context`, `AuthContextHeaderHandler`) from `drmcp.core.clients` to `drtools.core.auth`; exposed a single `resolve_token_from_headers()` entrypoint consumed by `get_sdk_client`
- Updated all `drmcp` tests to align with the `drtools` refactor: corrected import paths, exception types, async mock targets, and direct error propagation expectations
- Fixed `MCPToolConfig` loading `.env` independently of `MCPServerConfig(_env_file=None)`; introduced `_MCPToolConfigNoEnvFile` subclass so config default assertions reflect true code defaults

## 0.8.14
- Isolated publish secrets to an environment

## 0.8.13
- Added base agent for retrieving and storing memory

## 0.8.12
- Removed fastmcp dependency from drtools
- Fixed all unit tests to handle dict returns instead of ToolResult objects after refactoring
- Removed ToolResult dependencies from test assertions and mock setups
- Fixed import paths and lint errors across all test files
- Updated test expectations to work with plain dictionary responses from tools
- Updated helpers.py to use FastMCP's get_http_headers with safe import handling

## 0.8.11
- Made build_workflow async

## 0.8.10
- Added GitHub Actions workflow `integration.yml`: path-filtered **Integration Tests** job for drmcp (runs when `src/datarobot_genai/drmcp`, `src/datarobot_genai/drtools`, `setup.py`, `tests/drmcp/integration`, or the workflow file changes; aligned with the e2e workflow pattern)
- DRMCP integration/ETE: `tests/drmcp/stub_credentials.py` plus `ete_test_server.py` default stub `DATAROBOT_*` env so `task drmcp-integration` can start the MCP server without a real API token in `.env`; shared stub token constant wired from `tests/drmcp/conftest.py`
- `test_interactive.py`: read `DR_LLM_GATEWAY_MODEL` and optional `LLM_TEMPERATURE` into the LLM Gateway client config (consistent with ETE helpers)

## 0.8.9
- Pin LiteLLM to safe version to prevent exploit (see https://github.com/BerriAI/litellm/issues/24518)

## 0.8.8
- When constructing the agent card, prefer the DATAROBOT_PUBLIC_API_ENDPOINT over DATAROBOT_API_ENDPOINT, avoiding connection issues in onprem environments.

## 0.8.7
- Rework NAT AG-UI integration
- Do not return final response from NAT twice

## 0.8.6
- Locked upperbound for dragent dependencies (fastapi, starlette) to avoid compatibility issues
- Locked lowerbound for AG-UI because of a change in reasoning events validation

## 0.8.5
- Added logic to sync MCP server deployment associated metadata with items in MCP server.

## 0.8.4
- Added use case tools: list_use_cases and list_use_case_assets (MODEL-22810)
- Added ToolType registrations for use_case, vdb, code_execution, and optimization
- Added corresponding config fields and CLI options for new tool types

## 0.8.3
- Added AG-UI Events for Llamaindex

## 0.8.2
- Added model and deployment tools: get_model_details (with optional feature impact and ROC curve), is_eligible_for_timeseries_training, get_prediction_history (MODEL-22809)
- CODEOWNERS: MCP team owns drtools; default Buzok with MCP overrides (last match wins)

## 0.8.1
- Fixed issue with NAT profiler interacting with MCP tools for langgraph
- Enabled MCP and tool tests for all agents
- Implemented decorator `nat_tool` which allows registering a function in NAT with a single line

## 0.8.0
- Allowed configuring step adaptor
- Reorganized `dragent` to submodules
- Disabled step adaptor for custom DataRobot models to prevert reporting events twice
- Added example of producing a single (non-streaming) response

## 0.7.8
- Updated enum values under DataRobotMCPPromptCategory and DataRobotMCPToolCategory to make them align with accepted enums of DataRobot public API.
- Removed UNKNOWN enum value from DataRobotMCPPromptCategory, DataRobotMCPToolCategory, and DataRobotMCPResourceCategory.

## 0.7.7
- Forwarded `x-untrusted-*` headers alongside `x-datarobot-*` headers in NAT `extract_datarobot_headers_from_context()`

## 0.7.6
- Added base agent e2e example under `e2e-tests/dragent/base/` demonstrating how to extend `BaseAgent` directly with litellm

## 0.7.5
- Added `authenticated_a2a_client` function group to dragent, to authenticate all api calls including calls to `/.well-known/agent-card.json`.

## 0.7.4
- Allowed running nat agents with per-user workflows with drum

## 0.7.3
- Added `temperature` parameter support to LLM MCP clients (`BaseLLMMCPClient`, `DRLLMGatewayMCPClient`): read from config dict and forwarded to `chat.completions.create`
- Added `LLM_TEMPERATURE` env var support in `get_openai_llm_client_config()` and `get_dr_llm_gateway_client_config()` to control LLM temperature in acceptance tests
- Switched tool parameter matching in `ToolBaseE2E` from exact equality to subset matching (`_check_dict_params_match`) to reduce test flakiness

## 0.7.2
- Added `nvidia-nat-crewai` support with crewai >= 1.1.0 compatibility patches

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
