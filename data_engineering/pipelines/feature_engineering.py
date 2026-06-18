"""
This is runs the whole RAG ingestion pipeline
1) extract raw docs
2) cleaning
3) chunking
4) embedding
4) loading to vector DB

Both embeddings and clean tesxt documents are separately loaded for different
purposes: RAG queries and finetuning
"""

import json
from pathlib import Path

import data_engineering.feature_engineering as fe
from data_engineering.pipelines.pipeline_decorators import pipeline


def save_json(name: str, data: list, output_dir: Path = Path("artifacts")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    with path.open("w") as f:
        json.dump([item.model_dump() for item in data], f, indent=2, default=str)


@pipeline
def feature_engineering_pipeline(
    author_full_names: list[str], wait_for: str | list[str] | None = None
) -> None:
    raw_documents = fe.query_data_warehouse(author_full_names)
    save_json("raw_documents", raw_documents)

    cleaned_documents = fe.clean_documents(raw_documents)
    save_json("cleaned_documents", cleaned_documents)

    fe.load_to_vector_db(cleaned_documents)

    embedded_documents = fe.chunk_and_embed(cleaned_documents)
    save_json("embedded_documents", embedded_documents)

    fe.load_to_vector_db(embedded_documents)
