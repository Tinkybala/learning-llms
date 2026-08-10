"""Merge a LoRA adapter hosted on Hugging Face into its base model and push
the resulting full-weights model to another Hugging Face repo.

Usage:
    python -m finetuning.merge_and_push_lora \
        --adapter_repo <user>/<lora-adapter-repo> \
        --output_repo <user>/<merged-model-repo> \
        [--base_model <user>/<base-model-repo>] \
        [--private]
"""

import argparse
import os

from dotenv import load_dotenv


def main(
    adapter_repo: str,
    output_repo: str,
    base_model: str | None = None,
    private: bool = False,
) -> None:
    import torch
    from huggingface_hub import login
    from peft import AutoPeftModelForCausalLM, PeftConfig
    from transformers import AutoTokenizer

    load_dotenv()
    login(os.environ.get("HF_TOKEN"))

    # The adapter's config already points at the base model it was trained
    # on, so it only needs to be overridden if you want to merge onto a
    # different (e.g. re-uploaded) copy of the base model.
    if base_model is None:
        base_model = PeftConfig.from_pretrained(adapter_repo).base_model_name_or_path
    print(f"Loading adapter '{adapter_repo}' onto base model '{base_model}'")

    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_repo,
        base_model_name_or_path=base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_repo)

    print("Merging LoRA weights into the base model...")
    model = model.merge_and_unload()

    print(f"Pushing merged model to '{output_repo}'")
    model.push_to_hub(output_repo, private=private)
    tokenizer.push_to_hub(output_repo, private=private)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter and push the full model to the Hub")
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
        "--base_model",
        type=str,
        default=None,
        help="Override the base model to merge onto (defaults to the one recorded in the adapter's config)",
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
        base_model=args.base_model,
        private=args.private,
    )
