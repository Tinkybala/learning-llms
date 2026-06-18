from abc import ABC

from pydantic import UUID4

from data_engineering.feature_engineering.vector import VectorBaseDocument
from data_engineering.pipelines.ODM.types import DataCategory


class CleanedDocument(VectorBaseDocument, ABC):
    content: str
    platform: str
    author_id: UUID4
    author_full_name: str


class CleanedCustomArticleDocument(CleanedDocument):
    link: str

    class Config:
        name = "cleaned_custom_articles"
        category = DataCategory.ARTICLES
        use_vector_index = False


class CleanedRepositoryDocument(CleanedDocument):
    name: str
    link: str

    class Config:
        name = "cleaned_repositories"
        category = DataCategory.REPOSITORIES
        use_vector_index = False
