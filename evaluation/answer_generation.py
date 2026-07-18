# from tqdm.auto import tqdm
import gc

from vllm import LLM, SamplingParams

from datasets import load_dataset


def generate_answers(model_id, dataset_name):
    # Load dataset and format chat template
    dataset = load_dataset(dataset_name, split='test')

    def format(sample):
        return "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n".format(sample["instruction"])
    dataset = dataset.map(lambda sample: {"prompt": format(sample)})

    # initialise llm
    llm = LLM(model=model_id, max_model_len=4096)
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, min_p=0.05, max_tokens=4096)
    outputs = llm.generate(dataset['prompt'], sampling_params)

    answers = [output.outputs[0].text for output in outputs]
    dataset = dataset.add_column("answers", answers)
    print(f"Uploading results for {model_id}")
    dataset.push_to_hub(f"TInkybala/{model_id.split('/')[-1]}-results")
    gc.collect()
    return dataset

model_ids = [
    "TInkybala/llama-3.1-8b-finetune-test",
    "meta-llama/Llama-3.1-8B-Instruct"
]

for model_id in model_ids:
    generate_answers(model_id, "TInkybala/llmtwin_instruction_20260619_112900")
