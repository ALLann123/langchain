#!/usr/bin/python3
from IPython.display import Image, display
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
import random
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from pprint import pprint
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
model= ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)


#lets create a list of human and AI messages
messages=[AIMessage(content=f"SO you were researching ocean mammals?", name="model")]
messages.append(HumanMessage(content=f"Yes thats right.",name="Allan"))
messages.append(AIMessage(content=f"Great, what would you like to learn about.", name="model"))
messages.append(HumanMessage(content=f"I want to learn about the best place to see Orcas in the US", name="Allan"))

#lets iterate through our list and print the messages
for m in messages:
    m.pretty_print()

print("***************************************")
#lets invoke our model
llm=model

result=llm.invoke(messages)

print(f"AI: {result.content}")