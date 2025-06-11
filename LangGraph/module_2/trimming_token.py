#!/usr/bin/python3
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from pprint import pprint
from langchain_core.messages import trim_messages

load_dotenv()

load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
llm= ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

messages=[AIMessage(f" So you said you were researching ocean mammals?", name="Bot")]
messages.append(HumanMessage(f"Yes, I know about whales. But what others should I learn about?", name="Allan"))

# Node
def chat_model_node(state: MessagesState):
    messages = trim_messages(
            state["messages"],
            max_tokens=100,
            strategy="last",
            token_counter=ChatOpenAI(model="gpt-4o"),
            allow_partial=False,
        )
    return {"messages": [llm.invoke(messages)]}

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)
graph = builder.compile()

messages.append(HumanMessage(f"Tell me where Orcas live!", name="Allan"))


# Invoke, using message trimming in the chat_model_node 
messages_out_trim = graph.invoke({'messages': messages})