"""
Runs the retrieval pipeline including query expansion, self-query (metadata filtering),
vectorDB search and reranking
"""

import logging

from retrieval.context_retriever import ContextRetriever

logging.basicConfig(level=logging.INFO)

query = """
My name is Lilian Weng.

Can you write a draft for my blog explaining diffusion models?
I'm interested in some of the latest techniques that are used besides DDPM/DDIM.
"""

retriever = ContextRetriever(mock=False)
documents = retriever.search(query, k=3)

logging.info("Retrieved documents")
for rank, document in enumerate(documents):
    logging.info(f"{rank + 1}: {document}")
