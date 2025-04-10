#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

#load environment variables
load_dotenv()
api_key=os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

#get the user message
while True:
    get_message=input("YOU: ")
    #pass the message to our model
    results=model.invoke(get_message)
    print("-------------------------------------------------------------------------------")
    print("Results:")
    print(results.content)