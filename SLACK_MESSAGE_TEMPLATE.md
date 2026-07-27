## Slack Message for Commit Review Thread

**Format:** Slack Block Kit JSON for Slack message update

### Message Content (Human-Readable)

**Commit Review Thread - Documentation Updates Required**

PRs merged in the last 24 hours requiring documentation review:

🔗 [PR #593 - Revert PR 580 with ragas removal](https://github.com/datarobot-oss/datarobot-genai/pull/593)
_Significant API changes affecting how all frameworks (CrewAI, LangGraph, LlamaIndex) serialize pipeline interactions; LangGraph MCP API change (`load_mcp_tools(session=None, connection=...)`); README.md updated to reflect ragas dependency removal_

---

### Note
- PR #597 reviewed but does not require documentation updates (test-focused changes only)

---

### Slack Block Kit JSON

```json
{
  "type": "section",
  "text": {
    "type": "mrkdwn",
    "text": "*Commit Review Thread - Documentation Updates Required*\n\nPRs merged in the last 24 hours requiring documentation review:\n\n🔗 <https://github.com/datarobot-oss/datarobot-genai/pull/593|PR #593 - Revert PR 580 with ragas removal>\n_Significant API changes affecting how all frameworks (CrewAI, LangGraph, LlamaIndex) serialize pipeline interactions; LangGraph MCP API change (`load_mcp_tools(session=None, connection=...)`); README.md updated to reflect ragas dependency removal_"
  }
}
```

### Curl Command to Update Message (requires SLACK_WEBHOOK_URL)

```bash
curl -X POST \
  -H 'Content-type: application/json' \
  --data '{"text":"Commit Review Thread - Documentation Updates Required","blocks":[{"type":"section","text":{"type":"mrkdwn","text":"*Commit Review Thread - Documentation Updates Required*\n\nPRs merged in the last 24 hours requiring documentation review:\n\n🔗 <https://github.com/datarobot-oss/datarobot-genai/pull/593|PR #593 - Revert PR 580 with ragas removal>\n_Significant API changes affecting how all frameworks (CrewAI, LangGraph, LlamaIndex) serialize pipeline interactions; LangGraph MCP API change (`load_mcp_tools(session=None, connection=...)`); README.md updated to reflect ragas dependency removal_"}}]}' \
  $SLACK_WEBHOOK_URL
```
