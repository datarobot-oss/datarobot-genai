# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# A2A v0.3 backward-compatibility suite.
#
# Everything in this package pins what an A2A v0.3 caller or agent sees: the agent card, the
# JSON-RPC responses and the shapes dragent hands back to its own callers. The tests assert only
# on v0.3 behaviour, and the recorded documents live in ``golden/``.
#
# Rule for SDK upgrades: this suite must pass with its assertions and its ``golden/`` files left
# as they are. A change to either means a v0.3 caller would notice the upgrade, so it needs an
# explicit decision, not a quiet re-record. Imports and adapters may change to follow the SDK.
#
# Deliberately not pinned: the deprecated v0.3 routes ``GET /.well-known/agent.json`` and
# ``GET /agent/authenticatedExtendedCard``, which the SDK upgrade drops. The suite reaches the
# card only through ``GET /.well-known/agent-card.json`` and the JSON-RPC method
# ``agent/getAuthenticatedExtendedCard``, which stay. Keep deprecated routes out of this suite.
#
# Accepted differences from a2a-sdk 0.3 (a decision, not a re-record): the JSON-RPC error that
# answers ``message/send`` when the workflow fails carries ``"data": null``, which 0.3 left out.
# JSON-RPC 2.0 makes ``data`` optional, so ``golden/failing_rpc_message_send.json`` has it.
