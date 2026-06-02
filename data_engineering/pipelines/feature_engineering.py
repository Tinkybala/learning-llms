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

from 