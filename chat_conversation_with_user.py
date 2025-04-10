#!/usr/bin/python3
from dotenv import load_dotenv
import os
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

# Initialize chat history and system prompt
chat_history = []
system_message = SystemMessage(content="You are a helpful flower shop assistant")
chat_history.append(system_message)

# Chat loop
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break

    # Add user input
    chat_history.append(HumanMessage(content=query))

    # Get model response
    result = model.invoke(chat_history)
    response = result.content

    # Add AI response to history
    chat_history.append(AIMessage(content=response))

    print("-------------------------------------------------------------------")
    print(f"AI: {response}\n")

# Show final chat history
print("Message History Below:\n")
'''for msg in chat_history:
    role = "System" if isinstance(msg, SystemMessage) else "AI" if isinstance(msg, AIMessage) else "You"
    print(f"{role}: {msg.content}")'''
print(chat_history)
