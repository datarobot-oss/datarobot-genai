# GitHub MCP Tools

This module provides GitHub integration through tools proxied from GitHub's official MCP server at `https://api.githubcopilot.com/mcp/`.

## Architecture

Tools are registered from a JSON manifest at startup, following the same pattern as Jira, Confluence, and GDrive tools.

### Flow

```
Server startup
    │
    ▼
Load tools from github_tools.json manifest
    │
    ▼
Register all enabled tools with dr_mcp_extras()
    │
    ▼
Server running...
    │
    ▼
User calls github_create_issue(...)
    │
    ▼
get_github_access_token() → OAuth via get_access_token("github")
    │
    ▼
Proxy call to api.githubcopilot.com/mcp/
    │
    ▼
Return result
```

## Files

| File | Description |
|------|-------------|
| `register.py` | Registration logic - loads tools from manifest at startup |
| `tools.py` | Helper functions for tool execution |
| `github_tools.json` | Manifest of GitHub tools with schemas |
| `scripts/fetch_manifest.py` | Dev script to regenerate the manifest |

## Authentication

Tools authenticate via OAuth at execution time using `get_access_token("github")` from the core auth module. This follows the same pattern as other OAuth-enabled tools (Jira, Confluence, GDrive).

## Enabling/Disabling Tools

Edit `github_tools.json` and set `"enabled": false` for any tool you want to disable:

```json
{
  "tools": [
    {
      "name": "delete_repository",
      "enabled": false
    }
  ]
}
```

## Regenerating the Manifest

The manifest contains tool definitions fetched from GitHub's MCP server. To update it with the latest tools, you need a GitHub token with appropriate permissions:

```bash
GITHUB_TOKEN=xxx python -m datarobot_genai.drmcp.tools.github.scripts.fetch_manifest
```

This will:
1. Fetch all available tools from the GitHub MCP server
2. Preserve existing `enabled` status for known tools
3. Default new tools to `enabled: true`
4. Update `github_tools.json`

Note: This is a dev-time operation, not runtime. The server loads from the manifest at startup.

## Tool Naming

All GitHub tools are prefixed with `github_` to avoid naming conflicts:

- GitHub MCP tool: `get_me` → Registered as: `github_get_me`
- GitHub MCP tool: `create_issue` → Registered as: `github_create_issue`
