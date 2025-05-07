#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

#load our enviroment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

#create the model we are using
model= ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

#the message we want to invoke to the llm
msg=HumanMessage(content="Hello, who are you?", name="Allan")

#add it to a message list
message=[msg]

#now invoke the message above
result=model.invoke(message)
print(f"AI: {result.content}")