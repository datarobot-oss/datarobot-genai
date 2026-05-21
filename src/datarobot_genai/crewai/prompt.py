""" 
CrewAI has a different concept -- prompt template strings that you pass as plain strings to the Agent constructor:

system_template -- Custom system prompt template (uses {{ .System }} placeholder)
prompt_template -- Custom user/task prompt template (uses {{ .Prompt }} placeholder)

Example usage
from crewai import Agent
import datarobot as dr

datarobot_prompt_template = dr.genai.PromptTemplate.get(prompt_template_id)
datarobot_prompt_template_version = datarobot_prompt_template.get_latest_version()
datarobot_prompt_template_string = datarobot_prompt_template_version.to_fstring()

agent = Agent(
    role="Researcher",
    goal="Find relevant papers",
    backstory="An expert researcher",
    prompt_template=datarobot_prompt_template_string,
)
"""