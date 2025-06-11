#!/usr/bin/python3
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from pprint import pprint

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

"""
#lets invoke our model
result=llm.invoke(messages)
print(result)
"""
#lets begin to build our graph
#create a node
def chat_model_node(state:MessagesState):
    return {"messages": llm.invoke(state["messages"][-1:])}

#build graph
builder=StateGraph(MessagesState)
builder.add_node("chat_model", chat_model_node)
builder.add_edge(START, "chat_model")
builder.add_edge("chat_model", END)

graph=builder.compile()

#time to start our graph
output=graph.invoke({'messages':messages})
for m in output['messages']:
    m.pretty_print()
    