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

# LLM settings for the LlamaIndex sample

[LLM configuration (shared)](../llm.md) lists the environment variables.

In [`workflow.yaml`](../../e2e-tests/dragent/llamaindex/workflow.yaml) the sample uses:

```yaml
llms:
  datarobot_llm:
    _type: datarobot-llm-component
```

and **`workflow.llm_name: datarobot_llm`**. Routing follows the same four outcomes as other stacks.
