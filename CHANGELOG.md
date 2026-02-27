# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.5.16
- Improve `dragent`'s event handling

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
