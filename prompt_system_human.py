#!/usr/bin/python3
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

#load environment variables
load_dotenv()
api_key=os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

#here we write the prompt
print("------Prompt with system and human messages-------")
messages=[
    ("system", "You are a comedian who tells jokes about {topic}"),
    ("human", "Tell me {joke_count} jokes ")
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic":"hackers","joke_count":3})

#here we invoke the model
results=model.invoke(prompt)

print()
print(results.content)