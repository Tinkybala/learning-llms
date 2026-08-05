import logging

from data_engineering.feature_engineering.queries import Query
from retrieval.steps.reranking import Reranker

logging.basicConfig(level=logging.INFO)

query = Query.from_str("What is the capital of Singapore")
chunks = [
    "The capital of China is Beijing, a city with milleniums of history.",
    "The country of the Republic of Singapore only has one city called Singapore which is also its capital.",
    "Singapore is a beautiful city with lots of greenery",
    "Kuala Lumpur is the capital city of Malaysia",
]
reranker = Reranker(
    mock=True
)  # mock = True allows testing with strings instead of EmbeddedChunk objects
reranked_documents = reranker.generate(query=query, chunks=chunks, keep_top_k=4)
for rank, doc in enumerate(reranked_documents):
    logging.info(f"{rank}) {doc}")
