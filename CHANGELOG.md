# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.5.2

- Added "DR docs" tools: a tool for searching DataRobot Agentic AI docs and returning most relevant doc pages (includes title, URL, content) using TF-IDF, and a tool for fetching any DataRobot docs page. Note: only supported for English documentation, not Japanese.

## 0.5.1

- Added the CHANGELOG.md file
- GitHub Action to verify changelog entries in pull requests

## 0.5.0

- No dependencies installed by default
- Optional `extra` options were converted into dependency groups and require `uv` for installation
- Added `auth` dependency group for authentication utilities
- Pandas dependency moved from the `core` to the `drmcp` dependency group
