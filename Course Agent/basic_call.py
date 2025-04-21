#!/usr/bin/python3
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


#load environment variables
load_dotenv()
api_key=os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com",
)

def prompt_me(query):
    messages=[
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content=f"Here is the users query {query}. Summarize to three statements.")
    ]
    return messages

while True:
    query=input("USER: ")
    if query.lower()=="exit":
        print("[+]Shutting down...")
        break
    
    prompt_template_mine=prompt_me(query)
    result=model.invoke(prompt_template_mine)
    print()
    print(f"AI: {result.content}")
    print("-----------------------------------------------------------------")
    


"""

result=model.invoke(query)
print(f" USER: {query}")
print()
print(f"AI response: {result.content}")
"""


"""
ourse Agent> python .\basic_call.py
USER: Python hello world program

AI: 1. A Python "Hello, World!" program is a simple script that prints the phrase "Hello, World!" to the console.  
2. It serves as a basic introduction to Python programming for beginners.
3. The program is written using the `print()` function: `print("Hello, World!")`.
-----------------------------------------------------------------
USER: exit
[+]Shutting down...
"""