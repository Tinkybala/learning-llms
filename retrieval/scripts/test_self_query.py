import logging

from data_engineering.feature_engineering.queries import Query
from retrieval.steps.self_query import SelfQuery

logging.basicConfig(level=logging.INFO)

query = Query.from_str(
    "I am Liian Weng. Write an article about the best types of advanced RAG methods."
)
self_query = SelfQuery(mock=False)
query_with_metadata = self_query.generate(query)
logging.info(f"Extracted author_id: {query_with_metadata.author_id}")
logging.info(f"Extracted author_full_name: {query_with_metadata.author_full_name}")
