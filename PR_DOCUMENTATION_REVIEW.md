# PR Documentation Review - Last 24 Hours

**Review Period:** 2026-07-27 (last 24 hours)
**Review Date:** 2026-07-27 16:01 UTC

## Summary

2 PRs merged in the last 24 hours. 1 PR requires documentation updates.

---

## PRs Requiring Documentation Updates

### [PR #593](https://github.com/datarobot-oss/datarobot-genai/pull/593) - Revert PR 580 with ragas removal

**Reason:** *Significant API changes affecting how all frameworks (CrewAI, LangGraph, LlamaIndex) serialize pipeline interactions; LangGraph MCP API change (`load_mcp_tools(session=None, connection=...)`); README.md updated to reflect ragas dependency removal*

---

## PRs NOT Requiring Documentation Updates

### [PR #597](https://github.com/datarobot-oss/datarobot-genai/pull/597) - Disable litellm local cost map fetching to fix flakes in e2e tests

**Reason:** E2E test environment changes only; no product/API changes; no documentation-level impact

---

## Notes

- PR #597 is test-focused and does not require documentation updates beyond CHANGELOG (already handled)
- PR #593 already has CHANGELOG entry (v0.26.12) but may need additional implementation guide updates for the LangGraph MCP API changes
- Both PRs have completed their CHANGELOG updates as indicated in their PR bodies
