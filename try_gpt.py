#!/usr/bin/python3
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

#load environment variables
load_dotenv()

api_key=os.getenv("GITHUB_TOKEN")

#create llm
model=ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

llm=model

result=llm.invoke("Hello there?")

print(f"AI: {result.content}")


"""
python .\try_gpt.py
AI: Hello! 😊 How can I assist you today?
"""