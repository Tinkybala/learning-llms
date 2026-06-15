from abc import ABC, abstractmethod
from typing import Generic, TypeVar, cast

from feature_engineering.chunks import Chunk
from feature_engineering.embedded_chunks import EmbeddedChunk

ChunkT = TypeVar("ChunkT", bound=Chunk)
EmbeddedChunkT = TypeVar("EmbeddedChunkT", bound=EmbeddedChunk)

embedding_model = None

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
    
    @abstractmethod
    def map_model(self, data_model: ChunkT, embedding: list[float]) -> EmbeddedChunkT:
        pass

