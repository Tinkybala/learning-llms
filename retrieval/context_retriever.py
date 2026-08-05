import concurrent
import logging

from qdrant_client.models import FieldCondition, Filter, MatchValue

from data_engineering.feature_engineering.dispatchers.embedding_dispatcher import (
    EmbeddingDispatcher,
)
from data_engineering.feature_engineering.embedded_chunks import (
    EmbeddedChunk,
    EmbeddedCustomArticleChunk,
    EmbeddedRepositoryChunk,
)
from data_engineering.feature_engineering.queries import EmbeddedQuery, Query
from retrieval.steps.query_expansion import QueryExpansion
from retrieval.steps.reranking import Reranker
from retrieval.steps.self_query import SelfQuery

logging.basicConfig(level=logging.INFO)


def flatten(nested_list: list) -> list:
    """Flatten a list of lists into a single list."""

    return [item for sublist in nested_list for item in sublist]


class ContextRetriever:
    def __init__(self, mock: bool = False) -> None:
        self._query_expander = QueryExpansion(mock=mock)
        self._metadata_extractor = SelfQuery(mock=mock)
        self._reranker = Reranker(mock=mock)

    def search(
        self,
        query: str,
        k: int = 3,
        expand_to_n_queries: int = 3,
    ) -> list:
        query_model = Query.from_str(query)
        query_model = self._metadata_extractor.generate(query_model)
        logging.info(
            "Successfully extracted the author_id from the query",
            extra={"author_id": query_model.author_id},
        )

        # Query Expansion
        n_generated_queries = self._query_expander.generate(
            query_model, expand_to_n=expand_to_n_queries
        )
        logging.info(
            f"Query expanded to {n_generated_queries} queries.",
            extra={"num_queries": len(n_generated_queries)},
        )

        # Search
        with concurrent.futures.ThreadPoolExecutor() as executor:
            search_tasks = [
                executor.submit(self._search, _query_model, k)
                for _query_model in n_generated_queries
            ]

            n_k_documents = [
                task.result() for task in concurrent.futures.as_completed(search_tasks)
            ]
            n_k_documents = flatten(n_k_documents)
            n_k_documents = list(set(n_k_documents))
            logging.info(
                "All documents retrieved successfully.",
                extra={"num_documents": len(n_k_documents)},
            )

        # Re-Ranking
        if len(n_k_documents) > 0:
            k_documents = self.rerank(query, chunks=n_k_documents, keep_top_k=k)
        else:
            k_documents = []

        return k_documents

    def _search(self, query: Query, k: int = 3) -> list[EmbeddedChunk]:
        assert k >= 3, "k should be >= 3"

        def _search_data_category(
            data_category_odm: type[EmbeddedChunk], embedded_query: EmbeddedQuery
        ) -> list[EmbeddedChunk]:
            # filter by author_id
            if embedded_query.author_id:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="author_id",
                            match=MatchValue(value=str(embedded_query.author_id)),
                        )
                    ]
                )
            else:
                query_filter = None

            return data_category_odm.search(
                query_vector=embedded_query.embedding,
                limit=k // 2,  # //2 or //3 depend on how many document types wwe have
                query_filter=query_filter,
            )

        # Embed query and search
        embedded_query: EmbeddedQuery = EmbeddingDispatcher.dispatch(query)

        article_chunks = _search_data_category(
            EmbeddedCustomArticleChunk, embedded_query
        )
        repositories_chunks = _search_data_category(
            EmbeddedRepositoryChunk, embedded_query
        )
        retrieved_chunks = article_chunks + repositories_chunks

        return retrieved_chunks

    def rerank(
        self, query: str | Query, chunks: list[EmbeddedChunk], keep_top_k: int
    ) -> list[EmbeddedChunk]:
        if isinstance(query, str):
            query = Query.from_str(query)

        reranked_documents = self._reranker.generate(
            query=query, chunks=chunks, keep_top_k=keep_top_k
        )

        logging.info(
            "Documents reranked successfully.",
            extra={"num_documents": len(reranked_documents)},
        )
        return reranked_documents
