# Learning LLM

I'm using this repo to learn llm engineering.

## Setup
```
# To start MongoDB and Qdrant
docker-compose up -d

# Setup virtual environment
uv sync

# Install Unsloth
uv pip install unsloth --torch-backend=auto
```

## Running the code
```
# To run the etl pipeline 
poe run_lilian_etl

# To run the Vector DB ingestion pipeline
poe run_feature_engineering

```

## Training Data Generation & Training
```
# To generate instrcution-answer dataset
python finetuning/instruction_set_generation.py

# To run SFT training (see the file for more options)
python finetuning/finetuning_sft.py --qlora --dataset1 "TInkybala/llmtwin_instruction_20260619_112900" --epochs 3 --hub_model_id "TInkybala/llama-3.1-8b-finetune-test"
```

## Retrieval Steps
```
# To test query expansion (creating multiple rag queries to cover a larger search space)
poe test_query_expansion

# To test self-query (asking an llm to extract metadata to apply hard constraint filters on the db before doing search)
poe test_self_query

# To test the Cross-Encoder model
poe test_cross_encoder
```