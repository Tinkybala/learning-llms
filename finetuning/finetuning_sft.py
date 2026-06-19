import unsloth
import os
import torch
from trl import SFTTrainer
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments, TextStreamer
from unsloth import FastLanguageModel, is_bfloat16_supported
import argparse

def main(
    model_name: str = "meta-llama/Meta-Llama-3.1-8B",
    use_qlora: bool = False,
    lora_rank: int = 32,
    lora_alpha: int = 32,
    lora_dropout: float = 0,
    dataset1_name: str = None,
    lr: float = 3e-4,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    epochs: int = 1,
    hub_model_id: str = None
):  
    
    print("Training ")
    # Model
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=use_qlora,
    )

    # LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"],
    )

    # Dataset
    dataset1 = load_dataset(dataset1_name, split="train")
    # Supplementary general dataset; take 10k training samples
    dataset2 = load_dataset("mlabonne/FineTome-Alpaca-100k", split="train[:10000]") 
    dataset = concatenate_datasets([dataset1, dataset2])

    

    # Chat Template
    alpaca_template = """
        Below is an instruction that describes a task.
        Write a response that appropriately completes the request.

        ### Instruction:
        {}
        
        
        ### Response:
        {}"""
    
    
    EOS_TOKEN = tokenizer.eos_token

    def format_samples(examples):
        text = []
        for instruction, output in zip(examples["instruction"], examples["output"], strict=False):
            message = alpaca_template.format(instruction, output) + EOS_TOKEN
            text.append(message)

        return {"text": text}
    
    dataset = dataset.map(
        format_samples,
        batched=True,
        remove_columns=dataset.column_names,
        )
    
    # Validation
    dataset = dataset.train_test_split(test_size=0.05)

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            learning_rate=lr,
            lr_scheduler_type="linear",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs,
            fp16=not is_bfloat16_supported(),
            bf16= is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.1,
            warmup_steps=10,
            output_dir="output",
            push_to_hub=True if hub_model_id else False,
            hub_model_id = hub_model_id,
            report_to="wandb",
            seed=0
        )
    )

    trainer.train()





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT script")
    parser.add_argument("--qlora", action="store_true", help="Train with QLoRA")
    parser.add_argument("--dataset1", type=str, required=True, help="Name of the first dataset")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--hub_model_id", type=str, required=False, help="Huggingface model id to be saved to")

    args = parser.parse_args()
    
    main(use_qlora=args.qlora, dataset1_name=args.dataset1, epochs=args.epochs, hub_model_id=args.hub_model_id)
