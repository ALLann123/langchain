#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
#from IPython.display import Image, display # type: ignore

#lets set our llm
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

def multiply(a: int, b: int) -> int:
    """Multiply a and b
    Args:
        a: first int
        b: second int
    """ 
    return a *b

#this will be a tool
def add(a: int, b:int)-> int:
    """Adds a and b
    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int)-> float:
    """Divide a by b.
    Args:
        a: first int
        b: second int
    """
    return a/b

#tools are assigned in a list
tools=[add, multiply, divide]
llm=model
#bind the tools to our llm
llm_with_tools=llm.bind_tools(tools)

#system message
sys_msg=SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs")

#Node
def assistant(state: MessagesState):
    return {"messages":[llm_with_tools.invoke([sys_msg]+state["messages"])]}

#Graph
builder=StateGraph(MessagesState)

#define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

#define edges: these determine the control flow
builder.add_edge(START, "assistant")

builder.add_conditional_edges(
    "assistant",
    #If the latest message(result) from assistant is a tool call -> tools_condition routes to END
    #If the lates message(result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,   
)
builder.add_edge("tools", "assistant")

memory=MemorySaver()

graph=builder.compile(interrupt_before=["tools"], checkpointer=memory)

#show
#display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
#input 
initial_input={"messages": HumanMessage(content="Multiply 2 and 3")}

#create a thread
thread={"configurable": {"thread_id":"1"}}

#run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="values"):
    event['messages'][-1].pretty_print()

#get iser feedback
user_approval=input("Do you want to call the tool?(yes/no):")

#check approval
if user_approval.lower() == "yes":
    #if approved, continue the graph execution
    for event in graph.stream(None, thread, stream_mode="values"):
        event['messages'][-1].pretty_print()

else:
    print("[-]Operation not permitted by user.")


"""
python tool_breakpoint.py
================================ Human Message =================================
================================ Human Message =================================

Multiply 2 and 3
================================== Ai Message ==================================
Tool Calls:
  multiply (call_YsU32IiGkVXZZoAGkdtlsFsw)
 Call ID: call_YsU32IiGkVXZZoAGkdtlsFsw
  Args:
    a: 2
    b: 3
Do you want to call the tool?(yes/no):yes
================================== Ai Message ==================================
Tool Calls:
  multiply (call_YsU32IiGkVXZZoAGkdtlsFsw)
 Call ID: call_YsU32IiGkVXZZoAGkdtlsFsw
  Args:
    a: 2
    b: 3
================================= Tool Message =================================
Name: multiply

6
================================== Ai Message ==================================

The result of multiplying 2 and 3 is 6.
"""