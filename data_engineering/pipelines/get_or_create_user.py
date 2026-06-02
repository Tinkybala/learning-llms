import logging

from typing_extensions import Annotated

from data_engineering import utils
from data_engineering.pipelines.ODM.documents import UserDocument


def get_or_create_user(user_full_name: str) -> Annotated[UserDocument, "user"]:
    logging.info(f"Getting or creating user: {user_full_name}")
    first_name, last_name = utils.split_user_full_name(user_full_name)
    user = UserDocument.get_or_create(first_name=first_name, last_name=last_name)
    return user


def _get_metadata(user_full_name: str, user: UserDocument) -> dict:
    return {
        "query": {
            "user_full_name": user_full_name,
        },
        "retrieved": {
            "user_id": str(user.id),
            "first_name": user.first_name,
            "last_name": user.last_name,
        },
    }
