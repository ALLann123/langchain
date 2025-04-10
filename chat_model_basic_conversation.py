#!/usr/bin/python3
from dotenv import load_dotenv
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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

#now lets set the system message
#system message-Message for determining AI behaviour. Passed inthe first sequence of input message.

message= [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divide by 9?"),
]

#invoke the model with the message
result=model.invoke(message)

print(f"AI Response: {result.content}")

# AIMessage:
#   Message from an AI.
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")