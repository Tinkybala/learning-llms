from langchain_core.prompts import PromptTemplate

from retrieval.interface import PromptTemplateFactory
from retrieval.prompts import QUERY_EXPANSION_PROMPT, SELF_QUERY_PROMPT


class QueryExpansionTemplate(PromptTemplateFactory):

    prompt: str = QUERY_EXPANSION_PROMPT

    @property
    def seperator(self) -> str:
        return "#next-question#"
    
    def create_template(self, expand_to_n: int) -> PromptTemplate:
        return PromptTemplate(
            template=self.prompt,
            input_variables=["question"],
            partial_variables={
                "separator": self.seperator,
                "expand_to_n": expand_to_n,
            }
        )
    
class SelfQueryTemplate(PromptTemplateFactory):

    prompt: str = SELF_QUERY_PROMPT

    def create_template(self):
        return PromptTemplate(template=self.prompt, input_variables=["question"])