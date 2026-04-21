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

# Caveats

- **`POST /v1/chat/completions` with `stream=true`**: streamed chunks are **not** in the standard OpenAI Chat Completions delta shape—do not point a strict OpenAI client at streaming here unless you accept that mismatch.
- **Non-streaming `POST /generate`**: the JSON body may **not** match what a minimal “single AG-UI blob” client expects when the workflow behaves like a chat-completions response—validate against your client assumptions.
