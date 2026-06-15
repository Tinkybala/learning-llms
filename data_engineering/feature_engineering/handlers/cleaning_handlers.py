from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from feature_engineering.utils import clean_text

from data_engineering.feature_engineering.cleaned_document import (
    CleanedCustomArticleDocument,
    CleanedDocument,
    CleanedRepositoryDocument,
)
from data_engineering.pipelines.ODM.documents import (
    CustomArticleDocument,
    Document,
    RepositoryDocument,
)

DocumentT = TypeVar("DocumentT", bound=Document)
CleanedDocumentT = TypeVar("CleanedDocumentT", bound=CleanedDocument)


class CleaningDataHandler(ABC, Generic[DocumentT, CleanedDocumentT]):
    """
    Abstract class for all cleaning data handlers.
    All data transformations logic for the cleaning step is done here
    """

    @abstractmethod
    def clean(self, data_model: DocumentT) -> CleanedDocumentT:
        pass


class CustomArticleCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: CustomArticleDocument) -> CleanedCustomArticleDocument:
        valid_content = [content for content in data_model.content.values() if content]

        return CleanedCustomArticleDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(valid_content)),
            platform=data_model.platform,
            link=data_model.link,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
        )


class RepositoryCleaningHandler(CleaningDataHandler):
    def clean(self, data_model: RepositoryDocument) -> CleanedRepositoryDocument:
        return CleanedRepositoryDocument(
            id=data_model.id,
            content=clean_text(" #### ".join(data_model.content.values())),
            platform=data_model.platform,
            name=data_model.name,
            link=data_model.link,
            author_id=data_model.author_id,
            author_full_name=data_model.author_full_name,
        )
