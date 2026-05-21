from datarobot.models.genai.prompt_template import PromptTemplateVersion
from llama_index.core.prompts import PromptTemplate

def get_prompt_template(prompt_template_version: PromptTemplateVersion) -> PromptTemplate:
    return PromptTemplate(prompt_template_version.to_fstring())
