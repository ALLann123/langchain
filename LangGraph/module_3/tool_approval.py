#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt.tool_node import ToolNode as PrebuiltToolNode  # Changed import
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

# Load API key
load_dotenv()
model = ChatGroq(
    temperature=0.3,  # Lower temperature for more factual responses
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

# Tool functions
def multiply(a: int, b: int) -> int:
    """Multiply a and b"""
    return a * b

def add(a: int, b: int) -> int:
    """Add a and b"""
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a by b"""
    return a / b

# Register tools
tools = [add, multiply, divide]
llm_with_tools = model.bind_tools(tools)

# System prompt
sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs")

# Assistant node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Custom tool handler node
def tool_node(state: MessagesState):
    last_message = state["messages"][-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        for call in last_message.tool_calls:
            tool_name = call["name"]
            if tool_name == "multiply":
                user_input = input("Tool 'multiply' requested. Do you want to proceed? (yes/no): ")
                if user_input.lower() != "yes":
                    print("[-] Operation cancelled by user.")
                    exit(0)
    
    # Use PrebuiltToolNode instead of ToolNode
    tool_node = PrebuiltToolNode(tools)
    return tool_node.invoke(state)

# Build graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", tool_node)

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

memory = MemorySaver()
graph = builder.compile(interrupt_before=["tools"], checkpointer=memory)

# Run input
initial_input = {"messages": [HumanMessage(content="Add 900 and 299")]}
thread = {"configurable": {"thread_id": "1"}}

# Initial run (before interruption)
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

# Resume only if approved
user_approval = input("Do you want to continue and call the tool? (yes/no): ")
if user_approval.lower() == "yes":
    for event in graph.stream(None, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()
else:
    print("[-] Tool call aborted by user.")

"""
 python tool_approval.py
================================ Human Message =================================

Multiply 2 and 3
================================== Ai Message ==================================
Tool Calls:
  multiply (call_rrbs)
 Call ID: call_rrbs
  Args:
    a: 2
    b: 3
Do you want to continue and call the tool? (yes/no): yes
================================== Ai Message ==================================
Tool Calls:
  multiply (call_rrbs)
 Call ID: call_rrbs
  Args:
    a: 2
    b: 3
Tool 'multiply' requested. Do you want to proceed? (yes/no): yes
================================= Tool Message =================================
Name: multiply

6
================================== Ai Message ==================================

The result is 6.
"""