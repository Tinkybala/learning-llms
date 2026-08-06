QUERY_EXPANSION_PROMPT = """
You are an AI language model assistant. Your task is
to generate {expand_to_n}
different versions of the given user question to retrieve relevant
documents from a vector
database. By generating multiple perspectives on the user question,
your goal is to help
the user overcome some of the limitations of the distance-based
similarity search.
Provide these alternative questions separated by '{separator}'.
Original question: {question}
"""

SELF_QUERY_PROMPT = """
You are an AI language model assistant. Your task is
to extract information from a user question.
The required information that needs to be extracted is the user name
or user id.
Your response should consist of only the extracted user name (e.g.,
John Doe) or id (e.g. 1345256), nothing else.
If the user question does not contain any user name or id, you should
return the following token: none.
For example:
QUESTION 1:
My name is Donald Trump and I want a post about...
RESPONSE 1:
Donald Trump
QUESTION 2:
I want to write a post about...
RESPONSE 2:
none
QUESTION 3:
My user id is 1345256 and I want to write a post about...
RESPONSE 3:
1345256
User question: {question}
"""
