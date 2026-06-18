from abc import ABC

from pydantic import UUID4, Field

from data_engineering.feature_engineering.vector import VectorBaseDocument
from data_engineering.pipelines.ODM.types import DataCategory


class Chunk(VectorBaseDocument, ABC):
    content: str
    platform: str
    author_id: UUID4
    author_full_name: str
    document_id: UUID4
    metadata: dict = Field(default_factory=dict)


class CustomArticleChunk(Chunk):
    link: str

    class Config:
        category = DataCategory.ARTICLES


class RepositoryChunk(Chunk):
    name: str
    link: str

    class Config:
        category = DataCategory.REPOSITORIES
