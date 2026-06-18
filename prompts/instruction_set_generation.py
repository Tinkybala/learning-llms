GENERATION_PROMPT = """
Based on the following extract, generate five
instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context.
each answer \
must provide a relevant paragraph based on the information found in
the \
context. Only use concepts from the context to generate the
instructions. \
Instructions must never explicitly mention a context, a system, a
course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \
Example instruction: Explain the concept of an LLM Twin. \
Example answer: An LLM Twin is essentially an AI character that
mimics your writing style, personality, and voice. \
It's designed to write just like you by incorporating these elements
into a language model. \
The idea is to create a digital replica of your writing habits using
advanced AI techniques. \
Provide your response in JSON format with the following structure:
{{
"instruction_answer_pairs": [
{{"instruction": "...", "answer": "..."}},
...
]
}}
Extract:
{extract}
"""

GENERATION_MODEL = "gpt-4o-mini"
