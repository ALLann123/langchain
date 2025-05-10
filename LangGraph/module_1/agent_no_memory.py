#!/usr/bin/python3
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
llm= ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

#this is another tool
def add(a: int, b:int) -> int:
    """Adds a and b
    Args:
        a:first int
        b:second int"""
    return a+b

def divide(a: int, b: int) -> float:
    """Divide a and b
    Args:
        a:first int
        b:secont int
    """
    return a/b

#add our tools to a list
tools=[add, multiply, divide]

#parallel tool calling by default the model defaults to parallel tool calling for effieciency
llm_with_tools=llm.bind_tools(tools, parallel_tool_calls= False)


#********lets create our prompt to get the desired behaviour*****
#System Message
sys_msg=SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

print("[+]Lets create a node")
#node
def assistant(state: MessagesState):
    return {"messages":[llm_with_tools.invoke([sys_msg] + state["messages"])]}

print("[+]Creating Graph...\n")

#Graph
builder=StateGraph(MessagesState)

#define the nodes. Assistant makes tool calls, while ToolNode calls the tool
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

#define the edges which determine how control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    #If the latest message(result) from asssitant is a tool call -> tools_condition routes to tools
    #If the latest message(result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition,
)

builder.add_edge("tools", "assistant")
react_graph = builder.compile()

#lets pass in our quetion
messages=[HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5")]
print(f"Human Quize: {messages}")
messages=react_graph.invoke({"messages":messages})

#display
for m in messages['messages']:
    m.pretty_print()



"""
python agent_no_memory.py
[+]Lets create a node
[+]Creating Graph...

Human Quize: [HumanMessage(content='Add 3 and 4. Multiply the output by 2. Divide the output by 5', additional_kwargs={}, response_metadata={})]
================================ Human Message =================================

Add 3 and 4. Multiply the output by 2. Divide the output by 5
================================== Ai Message ==================================
Tool Calls:
  add (call_306SYfkcewrxD3BcdE155Vz6)
 Call ID: call_306SYfkcewrxD3BcdE155Vz6
  Args:
    a: 3
    b: 4
================================= Tool Message =================================
Name: add

7
================================== Ai Message ==================================
Tool Calls:
  multiply (call_LKkde1oyGpUJZVeBqUbqIw1Z)
 Call ID: call_LKkde1oyGpUJZVeBqUbqIw1Z
  Args:
    a: 7
    b: 2
================================= Tool Message =================================
Name: multiply

14
================================== Ai Message ==================================
Tool Calls:
  divide (call_3a5iFPMLwSPVALo2XaamdHWB)
 Call ID: call_3a5iFPMLwSPVALo2XaamdHWB
  Args:
    a: 14
    b: 5
================================= Tool Message =================================
Name: divide

2.8
================================== Ai Message ==================================

The final result after performing the calculations is 2.8.
"""