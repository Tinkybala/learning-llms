from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

from data_engineering.feature_engineering.chunks import (
    Chunk,
    CustomArticleChunk,
    RepositoryChunk,
)
from data_engineering.feature_engineering.embedded_chunks import (
    EmbeddedChunk,
    EmbeddedCustomArticleChunk,
    EmbeddedRepositoryChunk,
)
from data_engineering.feature_engineering.queries import EmbeddedQuery, Query
from models.embedding import EmbeddingModelSingleton

ChunkT = TypeVar("ChunkT", bound=Chunk)
EmbeddedChunkT = TypeVar("EmbeddedChunkT", bound=EmbeddedChunk)

embedding_model = EmbeddingModelSingleton()


class EmbeddingDataHandler(ABC, Generic[ChunkT, EmbeddedChunkT]):
    """
    ABC for embedding data handlers.
    All data transformation logic for embedding is done here
    """

    def embed(self, data_model: ChunkT) -> EmbeddedChunkT:
        return self.embed_batch([data_model])[0]

    def embed_batch(self, data_model: list[ChunkT]) -> list[EmbeddedChunkT]:
        embedding_model_input = [data_model.content for data_model in data_model]
        embeddings = embedding_model(embedding_model_input, to_list=True)

        embedded_chunk = [
            self.map_model(data_model, cast(list[float], embedding))
            for data_model, embedding in zip(data_model, embeddings, strict=False)
        ]
        return embedded_chunk

    @abstractmethod
    def map_model(self, data_model: ChunkT, embedding: list[float]) -> EmbeddedChunkT:
        pass


class RepositoryEmbeddingHandler(EmbeddingDataHandler):
    def map_model(
        self, data_model: RepositoryChunk, embedding: list[float]
    ) -> EmbeddedRepositoryChunk:
        return EmbeddedRepositoryChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            name=data_model.name,
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


class QueryEmbeddingHandler(EmbeddingDataHandler):
    def map_model(self, data_model: Query, embedding: list[float]) -> EmbeddedQuery:
        return EmbeddedQuery(
            id=data_model.id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            content=data_model.content,
            embedding=embedding,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )


class CustomArticleEmbeddingHandler(EmbeddingDataHandler):
    def map_model(
        self, data_model: CustomArticleChunk, embedding: list[float]
    ) -> EmbeddedCustomArticleChunk:
        return EmbeddedCustomArticleChunk(
            id=data_model.id,
            content=data_model.content,
            embedding=embedding,
            platform=data_model.platform,
            link=data_model.link,
            document_id=data_model.document_id,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
            metadata={
                "embedding_model_id": embedding_model.model_id,
                "embedding_size": embedding_model.embedding_size,
                "max_input_length": embedding_model.max_input_length,
            },
        )
