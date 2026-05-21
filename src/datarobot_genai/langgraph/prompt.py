from datarobot.models.genai.prompt_template import PromptTemplateVersion
from langchain_core.prompts import ChatPromptTemplate

def get_prompt_template(prompt_template_version: PromptTemplateVersion) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(prompt_template_version.to_fstring())