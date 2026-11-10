<!--
  ~ Copyright 2026 DataRobot, Inc. and its affiliates.
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# Tools and MCP (LangGraph sample)

## MCP behavior when enabled

DataRobot can attach **extra tools** (including from an MCP deployment) when the agent runs. In the graph sample, those appear **alongside** tools defined in Python. If only local tools are listed and the injected list is ignored, **MCP tools will not be available** to the model.

Practical rule: **merge** platform tools with local tools wherever tools bind to the graph (the e2e `myagent.py` shows the intended pattern).

## Configuration outside the repo

MCP server URL and credentials come from the **deployment / environment**, not from the LangGraph `workflow.yaml` in the minimal sample. The NAT workflow example adds **`function_groups`** in YAML instead; see [nat/mcp.md](../nat/mcp.md). With `MCP_WORKLOAD_ID`, the agent asks the platform where that workload is served (`GET /api/v2/workloads/<id>/`) and appends `/mcp` to the endpoint it reports, so the same variable works whether the cluster routes workloads through different API Gateway. The agent's API token therefore needs read access to the workload. If the lookup cannot answer, the agent runs without MCP tools and logs a warning. Then, no URL is guessed, because a composed one would be wrong on some clusters. Use `EXTERNAL_MCP_URL` to address a workload directly instead.

## Send the caller's Okta token to the MCP server

By default the agent sends its DataRobot API token to an MCP server set with `MCP_DEPLOYMENT_ID` or `MCP_SERVER_PORT`. Set `MCP_USE_XAA=true` to send the caller's Okta token instead:

- The agent takes the caller's access token from the `x-datarobot-external-access-token` request header and exchanges it with Okta cross-application access (XAA).
- It sends only the exchanged token, as `Authorization: Bearer`. It sends no DataRobot API token and forwards no request headers.
- It never falls back to the API token. If the exchange fails, the request fails.

| Setting | Description |
|---|---|
| `MCP_USE_XAA` | `true` turns the exchange on. The default is `false`. |
| `IDP_AGENT_ID`, `IDP_AGENT_PRIVATE_KEY_JWK` | The agent's Okta identity. Required when `MCP_USE_XAA` is `true`. |
| `MCP_XAA_TRUSTED_ISSUER`, `MCP_XAA_EXCHANGE_AUDIENCE`, `MCP_XAA_TOKEN_URL` | The MCP server's XAA parameters. Set all three, or none. |
| `MCP_XAA_SCOPES` | A comma-separated list of the scopes to request. When it is unset, the agent requests no scopes. |
| `MCP_XAA_TOKEN_AUDIENCE` | Optional. Sent to Okta as `resource`. |

When no `MCP_XAA_*` setting is set, the agent reads the parameters from the MCP server's `/.well-known/oauth-protected-resource` document, and requests the scopes that document lists as `scopes_supported`. On DataRobot 11.14 and older, the agent cannot read that document from an MCP server deployed as a custom model, so set the `MCP_XAA_*` settings. Use the values of the MCP server's own `MCP_XAA_*` settings.

Set `MCP_XAA_SCOPES` to the same list the server publishes as `scopes_supported`: the server's own `MCP_XAA_SCOPES`, plus any scopes its tools require. A DataRobot MCP server behind Okta typically needs `dr.impersonation`:

```sh
MCP_USE_XAA=true
MCP_XAA_TRUSTED_ISSUER=https://<org>.okta.com
MCP_XAA_EXCHANGE_AUDIENCE=https://<org>.okta.com/oauth2/<authorization server ID>
MCP_XAA_TOKEN_URL=https://<org>.okta.com/oauth2/<authorization server ID>/v1/token
MCP_XAA_SCOPES=dr.impersonation
```

Before you turn it on:

- A request without the caller's Okta token fails with an error. That includes chats in the playground and calls made with a DataRobot API key. The same happens when the token is expired or Okta refuses the exchange.
- `MCP_WORKLOAD_ID` is not covered yet. The agent logs a warning and sends the DataRobot API token to a workload MCP server, as before.
- `EXTERNAL_MCP_URL` is not affected. The agent sends only `EXTERNAL_MCP_HEADERS` to it.
- The agent gets the token once per request and uses it for the whole run. A token taken from the cache has at least five minutes left.
- The agent needs the `dragent` extra of `datarobot-genai`.
- Code that reads `MCPConfig.server_config` directly gets no credentials. Use `await mcp_config.authenticated_server_config()`, as `mcp_tools_context` does.

## Automated tests

`e2e-tests/dragent_tests/test_mcp.py` exercises MCP tool calls when either **`MCP_DEPLOYMENT_ID`** or **`MCP_WORKLOAD_ID`** is set and a tool-capable agent is configured.
