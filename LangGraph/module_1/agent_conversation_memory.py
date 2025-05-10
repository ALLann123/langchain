#!/usr/bin/python3
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

#use the memory saver checkpointer
memory=MemorySaver()

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

#when compiling our graph we add the memory checkpointer
react_graph = builder.compile(checkpointer=memory)

#now lets set the threadID
config={"configurable": {"thread_id":"1"}}

print("*************************"*4)
print("                                        AGENT MEMORY EXAMPLE     ")
print("*************************"*4)
print("Start chatting with the assistant (type 'exit' to quit):")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting chat.")
        break

    messages = [HumanMessage(content=user_input)]
    result = react_graph.invoke({"messages": messages}, config)

    # Only get the last assistant message
    final_message = result["messages"][-1]
    if hasattr(final_message, "content"):
        print("Assistant:", final_message.content)



"""
[+]Lets create a node
[+]Creating Graph...

****************************************************************************************************
                                        AGENT MEMORY EXAMPLE
****************************************************************************************************
Start chatting with the assistant (type 'exit' to quit):
You: hello, my name is Allan
Assistant: Hello, Allan! How can I assist you today?
You: what is my name, and who are you?
Assistant: Your name is Allan, and I am a helpful assistant here to assist you with arithmetic or any questions you have. How can I help you today?
You: what arithmetic tools do you have?
Assistant: I have three main arithmetic tools at your service:    

1. **Addition** - I can add two numbers for you.
2. **Multiplication** - I can multiply two numbers for you.       
3. **Division** - I can divide one number by another for you.     

If you have any calculations you need help with, just let me know!
You: muliply 4 with 8. Take the results and divide by 2. Finally, add 100 to the output
Assistant: The final result is **116**. Let me know if I can assist you further!
You: exit 
Exiting chat.
"""