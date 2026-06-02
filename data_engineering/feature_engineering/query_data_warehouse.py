import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing_extensions import Annotated

from data_engineering import utils
from data_engineering.pipelines.ODM.documents import (
    CustomArticleDocument,
    Document,
    NoSQLBaseDocument,
    RepositoryDocument,
    UserDocument,
)

logging.basicConfig(level=logging.INFO)


def query_data_warehouse(
    author_full_names: list[str],
) -> Annotated[list, "raw_documents"]:
    documents = []
    authors = []
    for author_full_name in author_full_names:
        logging.info(f"Querying data warehouse for user: {author_full_name}")

        first_name, last_name = utils.split_user_full_name(author_full_name)
        logging.info(f"first_name: {first_name}, last_name: {last_name}")
        user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
        authors.append(user)

        results = fetch_all_data(user)  # ?
        user_documents = [
            doc for query_result in results.values() for doc in query_result
        ]  # why got double loop
        documents.extend(user_documents)

    return documents


# multi-thread the data fetching across the document types
def fetch_all_data(user: UserDocument) -> dict[str, list[NoSQLBaseDocument]]:
    user_id = str(user.id)
    with ThreadPoolExecutor() as executor:
        futures ={
            executor.submit(__fetch_articles, user_id): "articles",
            #executor.submit(__fetch_posts, user_id): "posts",
            executor.submit(__fetch_repositories, user_id): "repositories",
        }

        results = {}
        # this does not process the futures in order!
        # stuff in the loop will become "non-blocking"
        # although the loop itself is still blocking the execution of the function
        # (if somethings taking really long in the loop, the function will wait for
        # loop to complete)
        for future in as_completed(futures):
            query_name = futures[future]
            try:
                results[query_name] = future.result()
            except Exception:
                logging.exception(f"'{query_name}' fetch request failed")
                results[query_name] = []
        
        return results

def __fetch_articles(user_id) -> list[NoSQLBaseDocument]:
    return CustomArticleDocument.bulk_find(author_id=user_id)


def __fetch_repositories(user_id) -> list[NoSQLBaseDocument]:
    return RepositoryDocument.bulk_find(author_id=user_id)


def __fetch_posts(user_id) -> list[NoSQLBaseDocument]:
    raise NotImplementedError


def _get_metadata(documents: list[Document]) -> dict:
    metadata = {
        "num_documents": len(documents),
    }
    for document in documents:
        collection = document.get_collection_name()
        if collection not in metadata:
            metadata[collection] = {}
        if "authors" not in metadata[collection]:
            metadata[collection]["authors"] = list()

        metadata[collection]["num_documents"] = (
            metadata[collection].get("num_documents", 0) + 1
        )
        metadata[collection]["authors"].append(document.author_full_name)

    for value in metadata.values():
        if isinstance(value, dict) and "authors" in value:
            value["authors"] = list(set(value["authors"]))

    return metadata
