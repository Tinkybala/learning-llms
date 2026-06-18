import logging

from data_engineering.feature_engineering.handlers.embedding_handlers import (
    CustomArticleEmbeddingHandler,
    EmbeddingDataHandler,
    QueryEmbeddingHandler,
    RepositoryEmbeddingHandler,
)
from data_engineering.feature_engineering.vector import VectorBaseDocument
from data_engineering.pipelines.ODM.types import DataCategory

logging.basicConfig(level=logging.INFO)


class EmbeddingHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> EmbeddingDataHandler:
        if data_category == DataCategory.QUERIES:
            return QueryEmbeddingHandler()
        elif data_category == DataCategory.ARTICLES:
            return CustomArticleEmbeddingHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryEmbeddingHandler()
        else:
            raise ValueError("Unsupported data type")


class EmbeddingDispatcher:
    factory = EmbeddingHandlerFactory

    @classmethod
    def dispatch(
        cls, data_model: VectorBaseDocument | list[VectorBaseDocument]
    ) -> VectorBaseDocument | list[VectorBaseDocument]:
        is_list = isinstance(data_model, list)
        if not is_list:
            data_model = [data_model]

        if len(data_model) == 0:
            return []

        data_category = data_model[0].get_category()
        assert all(
            data_model.get_category() == data_category for data_model in data_model
        ), "Data models must be of the same category."
        handler = cls.factory.create_handler(data_category)

        embedded_chunk_model = handler.embed_batch(data_model)

        if not is_list:
            embedded_chunk_model = embedded_chunk_model[0]

        logging.info(
            "Data embedded successfully.",
            #data_category=data_category,
        )

        return embedded_chunk_model
