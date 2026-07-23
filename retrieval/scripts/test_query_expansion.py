import logging

from data_engineering.feature_engineering.queries import Query
from retrieval.steps.query_expansion import QueryExpansion

logging.basicConfig(level=logging.INFO)

query = Query.from_str("Write an article about the best types of advanced RAG methods.")
query_expander = QueryExpansion(mock=False)
expanded_queries = query_expander.generate(query, expand_to_n=3)
for expanded_query in expanded_queries:
    logging.info(expanded_query.content)