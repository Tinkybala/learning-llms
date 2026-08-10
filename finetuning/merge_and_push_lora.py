"""Merge an Unsloth-trained LoRA adapter hosted on Hugging Face into its
base model and push the resulting full-weights model to another Hugging
Face repo.

Usage:
    python -m finetuning.merge_and_push_lora \
        --adapter_repo <user>/<lora-adapter-repo> \
        --output_repo <user>/<merged-model-repo> \
        [--private]
"""

import argparse
import os

from dotenv import load_dotenv


def main(
    adapter_repo: str,
    output_repo: str,
    private: bool = False,
) -> None:
    from huggingface_hub import login
    from unsloth import FastLanguageModel

    load_dotenv()
    login(os.environ.get("HF_TOKEN"))

    print(f"Loading LoRA adapter '{adapter_repo}'")
    # Unsloth detects that the repo holds an adapter (adapter_config.json)
    # and automatically resolves + loads the base model it was trained on.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_repo,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )

    print(f"Merging adapter and pushing full weights to '{output_repo}'")
    model.push_to_hub_merged(
        output_repo,
        tokenizer,
        save_method="merged_16bit",
        token=os.environ.get("HF_TOKEN"),
        private=private,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge an Unsloth LoRA adapter and push the full model to the Hub")
    parser.add_argument(
        "--adapter_repo",
        type=str,
        required=True,
        help="Hugging Face repo id of the LoRA adapter, e.g. 'user/llama-3.1-8b-lora'",
    )
    parser.add_argument(
        "--output_repo",
        type=str,
        required=True,
        help="Hugging Face repo id to push the merged model to, e.g. 'user/llama-3.1-8b-merged'",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Push the merged model as a private repo",
    )

    args = parser.parse_args()

    main(
        adapter_repo=args.adapter_repo,
        output_repo=args.output_repo,
        private=args.private,
    )
