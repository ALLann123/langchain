#!/usr/bin/python3
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

result=llm.invoke("What is the squere root of 49?")

print(result.content)

"""
langchain course>python chat_models_starter.py
The square root of 49 is **7**.

"""

