import logging

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

from settings import settings

logging.basicConfig(level=logging.INFO)


class MongoDatabaseConnector:
    _instance: MongoClient | None = None

    def __new__(cls, *args, **kwargs) -> MongoClient:
        if cls._instance is None:
            try:
                cls._instance = MongoClient(settings.DATABASE_HOST)
            except ConnectionFailure:
                logging.error("Could not connect to database: {e}")
                raise

        logging.info(
            f"Connection to MongoDB with URI successful: {settings.DATABASE_HOST}"
        )

        return cls._instance


connection = MongoDatabaseConnector()
