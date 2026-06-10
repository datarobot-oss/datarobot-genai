# Atlassian (Jira / Confluence) auth

Under **`AUTH_RESOLUTION_STRATEGY=http`**, Jira and Confluence use OAuth On-Behalf-Of (provider types
`jira` / `confluence`) with header fallbacks `x-datarobot-jira-access-token` and
`x-datarobot-confluence-access-token`. Cloud ID is resolved from
`https://api.atlassian.com/oauth/token/accessible-resources`.

Under **`AUTH_RESOLUTION_STRATEGY=config`**, set credentials on the server:

> **Personal agents only:** `config` is for single-user, personal setups. Do not use it on shared or multi-tenant agents—every user would share the same server-side Atlassian and API tokens from env/config.

| Variable | Required | Description |
|---|---|---|
| `ATLASSIAN_API_TOKEN` | Yes | OAuth access token **or** Atlassian API token |
| `ATLASSIAN_EMAIL` | For API tokens | Account email; when set, enables API token Basic auth |
| `ATLASSIAN_SITE_URL` | With email | Cloud site URL, e.g. `https://your-domain.atlassian.net` |

**Auth mode detection (config only):**

- **`ATLASSIAN_EMAIL` set** → API token Basic auth (`email:api_token`). Cloud ID comes from
  `{ATLASSIAN_SITE_URL}/_edge/tenant_info`. REST calls use
  `https://api.atlassian.com/ex/jira/{cloudId}/...` (or `/ex/confluence/{cloudId}/...`) with
  `Authorization: Basic ...`.
- **`ATLASSIAN_EMAIL` not set** → `ATLASSIAN_API_TOKEN` is treated as a static OAuth access token
  (Bearer). Cloud ID comes from the OAuth accessible-resources endpoint.

Example (API token for local scripts):

```bash
export AUTH_RESOLUTION_STRATEGY=config
export ATLASSIAN_API_TOKEN=your-atlassian-api-token
export ATLASSIAN_EMAIL=you@example.com
export ATLASSIAN_SITE_URL=https://your-domain.atlassian.net
```
