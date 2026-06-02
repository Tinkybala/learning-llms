from abc import ABC

from pydantic import UUID4, Field

from .base.nosql import NoSQLBaseDocument
from .types import DataCategory


class Document(NoSQLBaseDocument, ABC):
    content: dict
    platform: str
    author_id: UUID4 = Field(alias="author_id")
    author_full_name: str = Field(alias="author_full_name")


class RepositoryDocument(Document):
    name: str
    link: str

    class Settings:
        name = DataCategory.REPOSITORIES


class CustomArticleDocument(Document):
    link: str

    class Settings:
        name = DataCategory.ARTICLES


class UserDocument(NoSQLBaseDocument):
    first_name: str
    last_name: str

    class Settings:
        name = "users"

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
