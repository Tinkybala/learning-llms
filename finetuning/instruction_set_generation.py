import concurrent.futures
import datetime
import json
import os
import re
from typing import List, Tuple

from datasets import Dataset
from openai import OpenAI
from tqdm import tqdm

from prompts.instruction_set_generation import GENERATION_MODEL, GENERATION_PROMPT


def load_article_from_json(file_path: str) -> Dataset:
    with open(file_path, "r") as file:
        data = json.load(file)

    return Dataset.from_dict(
        {
            "id": [item["id"] for item in data],
            "content": [item["content"] for item in data],
            "platform": [item["platform"] for item in data],
            "author_id": [item["author_id"] for item in data],
            "author_full_name": [item["author_full_name"] for item in data],
            "link": [item["link"] for item in data],
        }
    )


def clean_text(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_substrings(
    dataset: Dataset, min_length: int = 1000, max_length: int = 2000
) -> List[str]:
    extracts = []
    sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"

    for article in dataset["content"]:
        cleaned_article = clean_text(article)
        sentences = re.split(sentence_pattern, cleaned_article)

        current_chunk = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                if len(current_chunk) >= min_length:
                    extracts.append(current_chunk.strip())
                current_chunk = sentence + " "

            if len(current_chunk) >= min_length:
                extracts.append(current_chunk.strip())

    return extracts


class InstructionAnswerSet:
    def __init__(self, pairs: List[Tuple[str, str]]):
        self.pairs = pairs

    @classmethod
    def from_json(cls, json_str: str) -> "InstructionAnswerSet":
        data = json.loads(json_str)
        pairs = [
            (pair["instruction"], pair["answer"])
            for pair in data["instruction_answer_pairs"]
        ]
        return cls(pairs)

    def __iter__(self):
        return iter(self.pairs)

def generate_instruction_answer_pairs(extract: str, client: OpenAI) -> List[Tuple[str,str]]:
    prompt = GENERATION_PROMPT.format(extract=extract)

    completion = client.chat.completions.create(
        model = GENERATION_MODEL,
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant who \
                generates instruction-answer pairs based on the given context. \
                Provide your response in JSON format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format = {"type" : "json_object"},
        max_tokens=1200,
        temperature=0.7
    )

    # extract generated instruct-answer pairs
    result = InstructionAnswerSet.from_json(completion.choices[0].message.content)

    return result.pairs


def create_instruction_dataset(
        dataset: Dataset, client: OpenAI, num_workers: int = 4
) -> Dataset:
    extracts = extract_substrings(dataset)
    instruction_answer_pairs = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_instruction_answer_pairs, extract, client)
                   for extract in extracts]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            instruction_answer_pairs.extend(future.result())
        
    instructions, answers = zip(*instruction_answer_pairs)
    return Dataset.from_dict(
        {"instruction": list(instructions), "output": list(answers)}
    )

if __name__ == "__main__":
    from dotenv import load_dotenv
    from huggingface_hub import login
    
    load_dotenv()
    login(os.environ.get("HF_TOKEN"))
    client = OpenAI()
    
    # Load raw data
    raw_dataset = load_article_from_json("artifacts/cleaned_documents.json")
    print("Raw Dataset:")
    print(raw_dataset.to_pandas())

    # Create instruction dataset
    instruction_dataset = create_instruction_dataset(raw_dataset, client)
    print("Instruction Dataset:")
    print(instruction_dataset.to_pandas())

    # Train/Test split and export
    filtered_dataset = instruction_dataset.train_test_split(test_size=0.1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filtered_dataset.save_to_disk(f"./datasets/instruction_{timestamp}")
    filtered_dataset.push_to_hub(f"TInkybala/llmtwin_instruction_{timestamp}")
    