#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt.tool_node import ToolNode as PrebuiltToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

# Load API key
load_dotenv()
model = ChatGroq(
    temperature=0.3,
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
            if tool_name == "divide":
                user_input = input("Division operation requested. Do you want to proceed? (yes/no): ")
                if user_input.lower() != "yes":
                    print("[-] Division operation cancelled by user.")
                    exit(0)
    
    # Use PrebuiltToolNode to handle all tool calls
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
# Remove the general interrupt_before parameter
graph = builder.compile(checkpointer=memory)

# Run input
initial_input = {"messages": [HumanMessage(content="Divide 90099 and 299")]}
thread = {"configurable": {"thread_id": "1"}}

# Stream the execution
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()


"""
 python .\multiply_break_only.py
================================ Human Message =================================

Divide 90099 and 299
================================== Ai Message ==================================
Tool Calls:
  divide (call_78er)
 Call ID: call_78er
  Args:
    a: 90099
    b: 299
Division operation requested. Do you want to proceed? (yes/no): yes
================================= Tool Message =================================
Name: divide

301.3344481605351
================================== Ai Message ==================================

The result of the division is 301.3344481605351.
"""