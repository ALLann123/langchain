#!/usr/bin/python3
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.tools import tool
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from IPython.display import Image, display
from langchain_core.tools import tool

load_dotenv()

api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
model= ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

llm=model
#lets write our first tool
def multiply(a: int, b:int) -> int:
    """Multiply a and b.
    Args:
    a:first int
    b:second int
    """
    return a*b
llm_with_tools=llm.bind_tools([multiply])



#lets begin building our Graph
class MessagesState(MessagesState):
    pass

#lets create our tool calling node
def tool_calling_llm(state:MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

#build graph
builder=StateGraph(MessagesState)

#add the nodes
builder.add_node("tool_calling_llm", tool_calling_llm)

#connect the nodes
builder.add_edge(START, "tool_calling_llm")
builder.add_edge("tool_calling_llm", END)

app=builder.compile()

messages=app.invoke({"messages":HumanMessage(content="Multiple 3 and 5")})
print(messages)


for m in messages["messages"]:
    m.pretty_print()

