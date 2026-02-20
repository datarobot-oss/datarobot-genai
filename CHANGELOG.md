# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).


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
