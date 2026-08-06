import logging

from data_engineering.feature_engineering.embedded_chunks import EmbeddedChunk
from data_engineering.feature_engineering.queries import Query
from models.cross_encoder import CrossEncoderModelSingleton
from retrieval.interface import RAGStep

logging.basicConfig(level=logging.INFO)


class Reranker(RAGStep):
    def __init__(self, mock: bool = False) -> None:
        super().__init__(mock=mock)
        self._model = CrossEncoderModelSingleton()

    def generate(
        self, query: Query, chunks: list[EmbeddedChunk] | list[str], keep_top_k: int
    ) -> list[EmbeddedChunk]:
        """
        mock: set to True to test the cross encoder on strings instead of chunks
        """
        if self._mock:
            logging.info("Reranker: Running in mock")
            assert isinstance(chunks[0], str), (
                "str not provided for Reranker chunk in mock mode"
            )
            query_doc_tuples = [(query.content, content) for content in chunks]
        else:
            assert isinstance(chunks[0], EmbeddedChunk), (
                "list[EmbeddedChunk] not provided for Reranker chunk"
            )
            query_doc_tuples = [(query.content, chunk.content) for chunk in chunks]
        scores = self._model(query_doc_tuples)
        scored_query_doc_tuples = list(zip(scores, chunks, strict=False))
        scored_query_doc_tuples.sort(key=lambda x: x[0], reverse=True)
        reranked_documents = scored_query_doc_tuples[:keep_top_k]
        reranked_documents = [doc for _, doc in reranked_documents]

        return reranked_documents
