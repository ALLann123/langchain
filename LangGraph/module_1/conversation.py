#!/usr/bin/python3
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END

load_dotenv()

api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

llm = model

# Define the tool
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers a and b."""
    return a * b

# Bind tool with tool_choice="auto" (IMPORTANT)
llm_with_tools = llm.bind_tools([multiply], tool_choice="auto")

# Build the graph
class ChatbotState(MessagesState):
    pass

# Define the tool-calling node
def tool_calling_llm(state: ChatbotState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}

# Build and compile the graph
builder = StateGraph(ChatbotState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)

app = builder.compile()

# Initialize conversation history
chat_history = []

print("Welcome to your shell chatbot! (Type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    chat_history.append(HumanMessage(content=user_input))

    result = app.invoke({"messages": chat_history})

    last_bot_message = result["messages"][-1]
    print(f"Bot: {last_bot_message.content}")

    chat_history.append(last_bot_message)
