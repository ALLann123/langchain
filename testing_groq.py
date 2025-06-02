#!/usr/bin/python3
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

#set up the model
llm=ChatGroq(
    model="llama3-70b-8192", 
    api_key=os.getenv("GROQ_API_KEY")
)

result=llm.invoke("Write a python script to add numbers 12 and 14")

print(f"AI Feedback: {result.content}")


"""
python testing_groq.py
AI Feedback: Here is a Python script that adds the numbers 12 and 14:
```
# This is a comment - anything after the "#" symbol is ignored by the interpreter

# Add the numbers 12 and 14
result = 12 + 14

# Print the result
print("The result is:", result)
```
Save this code to a file with a `.py` extension (e.g. `add_numbers.py`) and run it using Python (e.g. `python add_numbers.py`) to see the output.

Alternatively, you can also use the interactive Python shell to execute this code:
```
$ python
 result = 12 + 14
 print("The result is:", result)
The result is: 26
```
Let me know if you have any questions!
"""