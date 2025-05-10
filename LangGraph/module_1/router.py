from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import os

print("Starting APP...\n")

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
llm= ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

llm_with_tools = llm.bind_tools([multiply])

# Define the node logic
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))
builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)

print("Creating Graph...")
graph = builder.compile()

# Run the graph
messages = [HumanMessage(content="Hello, what is 2 multiplied by 2?")]
messages = graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()



"""
 python router.py
Starting APP...

Creating Graph...
================================ Human Message =================================

Hello, what is 2 multiplied by 2?
================================== Ai Message ==================================
Tool Calls:
  multiply (call_hA7oaTyW1H9zpUWWs2yVfckA)
 Call ID: call_hA7oaTyW1H9zpUWWs2yVfckA
  Args:
    a: 2
    b: 2
================================= Tool Message =================================
Name: multiply

4
"""