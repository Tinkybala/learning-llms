import os

from dotenv import load_dotenv

load_dotenv()


import logging

from langchain_openai import ChatOpenAI

from data_engineering import utils
from data_engineering.feature_engineering.queries import Query
from data_engineering.pipelines.ODM.documents import UserDocument
from retrieval.interface import RAGStep
from retrieval.prompt_template import SelfQueryTemplate
from settings import settings

logging.basicConfig(level=logging.INFO)

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

class SelfQuery(RAGStep):
    def generate(self, query: Query) -> Query:
        if self._mock:
            logging.info("Running mock self query")
            query.author_id = "mock_id"
            query.author_full_name = "mock_username"
            return query
    
        prompt = SelfQueryTemplate().create_template()
        model = ChatOpenAI(model=settings.OPENAI_MODEL_ID, api_key=OPENAI_API_KEY, temperature=0)
        chain = prompt | model

        response = chain.invoke({"question": query})
        user_full_name = response.content.strip("\n")
        if user_full_name == "none":
            logging.info("Self Query: No user_full_name found")
            return query
    
        first_name, last_name = utils.split_user_full_name(user_full_name)
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
        query.author_id = user.id
        query.author_full_name = user.full_name
        return query