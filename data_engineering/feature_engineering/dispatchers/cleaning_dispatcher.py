import logging

from data_engineering.feature_engineering.vector import VectorBaseDocument
from data_engineering.pipelines.ODM.base.nosql import NoSQLBaseDocument
from data_engineering.pipelines.ODM.types import DataCategory

logging.basicConfig(level=logging.INFO)


class CleaningHandlerFactory:
    @staticmethod
    def create_handler(data_category: DataCategory) -> CleaningDataHandler:
        if data_category == DataCategory.ARTICLES:
            return CustomArticleCleaningHandler()
        elif data_category == DataCategory.REPOSITORIES:
            return RepositoryCleaningHandler()
        else:
            raise ValueError("Unsupported data type")

class CleaningDispatcher:
    cleaning_factory = CleaningHandlerFactory

    @classmethod
    def dispatch(cls, data_model: NoSQLBaseDocument) -> VectorBaseDocument:
        data_category = DataCategory(data_model.get_collection_name())
        handler = cls.cleaning_factory.create_handler(data_category)
        clean_model = handler.clean(data_model)

        logging.info(
            "data cleaned successfully."
        )
        return clean_model



    

