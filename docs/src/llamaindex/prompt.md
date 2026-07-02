

# Using DataRobot Prompt Templates with LlamaIndex

The `datarobot_genai.llama_index.prompt` module converts a DataRobot prompt template version into a LlamaIndex `PromptTemplate` so you can use centrally managed prompts in your agents.

## Fetch and convert

```python
import datarobot as dr
from datarobot_genai.llama_index.prompt import get_prompt_template

prompt_template = dr.genai.PromptTemplate.get("prompt_template_id")
prompt_template_version = prompt_template.get_latest_version()
prompt_template_from_dr = get_prompt_template(prompt_template_version)
```

`get_prompt_template` takes a `PromptTemplateVersion` and returns a LlamaIndex `PromptTemplate` with the template string already set.

## Use as a system prompt in a FunctionAgent (most prefarable variant, state_prompt is not reliable)

Pass the formatted template to `system_prompt` when creating your agent:

```python
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from datarobot_genai.llama_index.llm import get_llm

llm = get_llm(model_name="provider/llm")

agent = FunctionAgent(
    name="researcher",
    description="Answers research questions",
    system_prompt=prompt_template_from_dr.format(domain="machine learning"),
    llm=llm,
)

workflow = AgentWorkflow(agents=[agent], root_agent="researcher")
```