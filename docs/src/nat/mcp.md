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

# MCP tools in NAT workflows

For how to configure and connect MCP servers, see **MCP servers** in the agent
application template's docs. This page documents only what this library adds to NAT.

## `_type: datarobot_mcp_client`

NAT's `mcp_client` with one added field and one relaxed rule:

- **`server.name`** — which configured server the block connects to. The address is not in
  the YAML; it comes from that server's `<name>_MCP_*` variables. Defaults to `default`,
  the server the single-server variables configure, so a block that omits it keeps working.
- `server.url` is optional. NAT requires it; here an inline URL is the other way to address
  a server, and setting both an inline URL and environment variables for one name is an
  error rather than a precedence rule.

Everything else on the block — `include`, `exclude`, `tool_overrides`, the timeouts, the
`reconnect_*` family, `session_aware_tools`, `max_sessions`, `session_idle_timeout` — is
NAT's and behaves as
[NAT documents it](https://docs.nvidia.com/nemo/agent-toolkit/latest/build-workflows/mcp-client.html).

## `_type: datarobot_mcp_auth`

The auth provider for DataRobot-hosted servers. Per request it sends forwarded
`x-datarobot-*` headers, `Authorization: Bearer`, `x-datarobot-api-key` and the
authorization context.

NAT hands **one instance to every block naming it**, which is correct here because it
produces the same credentials for every server. A server reached without a DataRobot
identity names `auth_provider: none` and never reaches this provider.

`headers:` on this block is merged last and wins. It is the only way to attach a static
header to a DataRobot-hosted server.

## Behaviour

Clients are built once, at workflow build, and stay connected. A server that is configured
and unreachable fails the build, naming the server, rather than yielding an agent with no
tools.
