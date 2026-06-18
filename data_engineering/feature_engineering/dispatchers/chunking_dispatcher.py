import logging

from feature_engineering.handlers.chunking_handlers import (
    ChunkingDataHandler,
    CustomArticleChunkingHandler,
    RepositoryChunkingHandler,
)

from data_engineering.feature_engineering.vector import VectorBaseDocument
from data_engineering.pipelines.ODM.base.nosql import NoSQLBaseDocument
from data_engineering.pipelines.ODM.types import DataCategory

logging.basicConfig(level=logging.INFO)


class ChunkingHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> ChunkingDataHandler:
        if data_category == DataCategory.ARTICLES:
            return CustomArticleChunkingHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryChunkingHandler()
        else:
            raise ValueError("Unsupported data type")


class ChunkingDispatcher:
    chunking_factory = ChunkingHandlerFactory

    @classmethod
    def dispatch(cls, data_model: NoSQLBaseDocument) -> VectorBaseDocument:
        data_category = DataCategory(data_model.get_collection_name())
        handler = cls.chunking_factory.create_handler(data_category)
        chunk_models = handler.chunk(data_model)
        logging.info("Document chunked successfully")
        return chunk_models
