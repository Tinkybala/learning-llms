import logging

from typing_extensions import Annotated

from data_engineering import utils
from data_engineering.feature_engineering.vector import VectorBaseDocument

logging.basicConfig(level=logging.INFO)


def load_to_vector_db(
    documents: Annotated[list, "documents"],
) -> Annotated[bool, "success"]:
    logging.info(f"Loading {len(documents)} documents into the vector database")
    grouped_documents = VectorBaseDocument.group_by_class(documents)
    for document_class, documents in grouped_documents.items():
        logging.info(f"Loading documents into {document_class.get_collection_name()}")
        for documents_batch in utils.batch(documents, batch_size=4):
            try:
                document_class.bulk_insert(documents_batch)
            except Exception:
                logging.info(
                    f"Failed to insert documents into {document_class.get_collection_name()}"
                )
                return False
    return True
