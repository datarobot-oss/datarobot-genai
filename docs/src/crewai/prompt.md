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

# Using DataRobot Prompt Templates with CrewAI

CrewAI does not have a dedicated prompt template class. Instead, prompts are plain strings passed directly to `Agent` attributes like `backstory`, `goal` or `Task` attributes `description` or `expected_output`

| Where  | Field             | Best for                                 |
|--------|-------------------|------------------------------------------|
| Agent  | `backstory`       | Personality, role context, system prompt  |
| Agent  | `goal`            | Short objective/directive                 |
| Task   | `description`     | Step-by-step task instructions            |
| Task   | `expected_output` | Output format constraints                 |


## Example with task description

```python
# 1. Fetch and convert
import datarobot as dr

prompt_template = dr.genai.PromptTemplate.get("prompt_template_id")
prompt_template_version = prompt_template.get_latest_version()
prompt_text = prompt_template_version.to_fstring()

# 2. Build agents with the DR-managed prompt
llm = get_llm(model_name="provider/llm")

agent_writer = Agent(
    role="Content Writer",
    goal="Write a 2-3 sentence about: {topic}.",
    backstory=make_system_prompt(
        "You are a concise writer, write a short response."
    ),
    llm=llm,
)

agents = [agent_writer]

# 3. Define task
task_writer = Task(
    description=prompt_text
    expected_output="A concise 2-3 sentence response.",
    agent=agent_writer,
)

tasks = [task_writer]

# 4. Assemble and register
crew = Crew(agents=agents, tasks=tasks, stream=True)

kickoff_inputs = lambda user_prompt_content: {
    "topic": user_prompt_content,
    "chat_history": "",
}
```
