import os

from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI

from data_engineering.feature_engineering.queries import Query
from retrieval.interface import RAGStep
from retrieval.prompt_template import QueryExpansionTemplate
from settings import settings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class QueryExpansion(RAGStep):
    def generate(self, query: Query, expand_to_n: int) -> list[Query]:
        assert expand_to_n > 0, f"expand_to_n should be > 0. Got {expand_to_n}"

        if self._mock:
            return [query for _ in range(expand_to_n)]

        query_expansion_template = QueryExpansionTemplate()
        prompt = query_expansion_template.create_template(expand_to_n - 1)
        model = ChatOpenAI(
            model=settings.OPENAI_MODEL_ID, api_key=OPENAI_API_KEY, temperature=0
        )
        chain = prompt | model
        response = chain.invoke({"question": query})
        result = response.content

        queries_content = result.strip().split(query_expansion_template.seperator)
        queries = [query]
        queries += [
            query.replace_content(stripped_content)
            for content in queries_content
            if (stripped_content := content.strip())
        ]
        return queries
