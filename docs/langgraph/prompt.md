# Using DataRobot Prompt Templates with LangGraph

The `datarobot_genai.langgraph.prompt` module converts a DataRobot prompt template version into a LangChain `ChatPromptTemplate` so you can use centrally managed prompts in your agents.

## Usage

```python
import datarobot as dr
from datarobot_genai.langgraph.prompt import get_prompt_template

prompt_template = dr.genai.PromptTemplate.get("prompt_template_id")
prompt_template_version = prompt_template.get_latest_version()
prompt_template_from_dr = get_prompt_template(prompt_template_version)
```

`get_prompt_template` takes a `PromptTemplateVersion` and returns a `ChatPromptTemplate` with the template string already set.

## Use as a top-level prompt in a ChatPromptTemplate

Combine the DataRobot template with other message roles to build the full prompt. If the DataRobot template contains variables (e.g. `{topic}`), supply them at format

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
   ("system", prompt_template_from_dr.format(topic="topic")),
    ("placeholder", "{chat_history}"),
    ("user", "{topic}"),
])
```

## Use as a system_prompt inside individual graph nodes

Combine the DataRobot template with other message roles to build the full prompt.

```python

agent = create_agent(
    llm,
    tools=tools,
    system_prompt=prompt_template_from_dr.format(),  # formatted to string
    name="planner",
)
```

