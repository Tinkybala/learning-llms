from unsloth import FastLanguageModel
from transformers import TextStreamer

alpaca_template = """
        Below is an instruction that describes a task.
        Write a response that appropriately completes the request.

        ### Instruction:
        {}
        
        
        ### Response:
        {}"""

print("Loading model weights... (This will only happen once!)")

# 1. Load the model and tokenizer OUTSIDE the loop
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="/home/tinkybala/learning-llms/output/checkpoint-3045",
    max_seq_length=2048,
    dtype=None, 
    load_in_4bit=True
)

FastLanguageModel.for_inference(model)

# Set up the text streamer to print the response in real-time
text_streamer = TextStreamer(tokenizer, skip_prompt=True)

print("\nModel loaded successfully! Type 'exit' or 'quit' to stop.")
print("-" * 50)

# 2. Start the interactive testing loop
while True:
    # Get user input from the terminal
    instruction = input("\nUser: ")
    
    # Allow a way to break out of the loop gracefully
    if instruction.lower() in ["exit", "quit"]:
        print("Exiting...")
        break
        
    # Skip empty inputs
    if not instruction.strip():
        continue

    # Format the prompt
    prompt = alpaca_template.format(instruction, "")
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    print("\nModel: ", end="")
    
    # Generate the output (using the streamer so we don't need to manually decode/print)
    _ = model.generate(
        **inputs,
        streamer=text_streamer, 
        max_new_tokens=512,
        use_cache=True # Speeds up generation
    )
    print("\n" + "-" * 50)