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

# LLM settings for the LangGraph sample

## Environment

Use the same variables as every other sample: [LLM configuration (shared)](../llm.md). They control gateway vs deployment vs NIM vs external routing.

## In `workflow.yaml`

The LangGraph e2e file declares:

```yaml
llms:
  datarobot_llm:
    _type: datarobot-llm-component
```

**`workflow.llm_name`** must match that key. The component follows env-based routing; you normally do not duplicate ids in YAML unless your deployment needs extra headers.

## In Python (optional local runs)

When you run framework code outside DRAgent, the same routing applies if you construct an LLM the way the sample does—still driven by env.
