#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool, Tool

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

#function for the tools
def greet_user(name: str) -> str:
    #greets the user by name
    return f"Hello, {name}!"

def reverse_string(text: str)-> str:
    #reverse the given string
    return text[::-1]

def concantenate_strings(a: str, b: str) -> str:
    #combine two strings
    return a + b

#pydantic model for tool arguments
class ConcatenateStringsArgs(BaseModel):
    a: str=Field(description="First String")
    b: str=Field(description="Second string")


#create tools using Tool and StructuredTool constructor approach
tools=[
    #use Tool for simpler functions with a single input parameter
    #This is straight forward and does not require an input schema
    Tool(
        name="GreetUser", #name the tool
        func=greet_user, #the function to execute
        description="Greet the user by name.", #description of the tool
    ),
    #use tool for another simple input parameter
    Tool(
        name="ReverseString", 
        func=reverse_string,
        description="Reverse the given string",
    ),
    #use structuredtool for more complex functions that require multiple inputs
    #StructuredTool allows us to define the input schema using Pydantic, ensuring proper validation and description
    StructuredTool.from_function(
        func=concantenate_strings, #function to execute
        name="ConcantenateStrings", #name the tool
        description="Concantenates two strings.",
        args_schema=ConcatenateStringsArgs, #schema defining the tool's input arguments
    ),
]

# Initialize a chatOpenAI model
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com",
    temperature=0.3
)

#pull the prompt template from the hub
prompt = hub.pull("hwchase17/openai-tools-agent")

#create the ReAct agent using the create_tool_calling_agent function

agent =  create_tool_calling_agent(
    llm=llm,  #model to use
    tools=tools, #list of tools available to the agent
    prompt=prompt, #prompt template to guide the agents response
)

#create the agent executor
agent_executor=AgentExecutor.from_agent_and_tools(
    agent=agent, #the agent to execute
    tools=tools, #list of tools available to the agent
    verbose=True, 
    handle_parsing_errors=True,
)

# Test the agent with sample queries
response = agent_executor.invoke({"input": "Greet Alice"})
print("Response for 'Greet Alice':", response)

response = agent_executor.invoke({"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", response)

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world':", response)


"""

> Entering new AgentExecutor chain...

Invoking: `GreetUser` with `Alice`


Hello, Alice!Hello, Alice!

> Finished chain.
Response for 'Greet Alice': {'input': 'Greet Alice', 'output': 'Hello, Alice!'}


> Entering new AgentExecutor chain...

Invoking: `ReverseString` with `hello`


ollehThe reversed string of 'hello' is 'olleh'.

> Finished chain.
Response for 'Reverse the string hello': {'input': "Reverse the string 'hello'", 'output': "The reversed string of 'hello' is 'olleh'."}


> Entering new AgentExecutor chain...

Invoking: `ConcantenateStrings` with `{'a': 'hello', 'b': 'world'}`


Invoking: `ConcantenateStrings` with `{'a': 'hello', 'b': 'world'}`

Invoking: `ConcantenateStrings` with `{'a': 'hello', 'b': 'world'}`
Invoking: `ConcantenateStrings` with `{'a': 'hello', 'b': 'world'}`
Invoking: `ConcantenateStrings` with `{'a': 'hello', 'b': 'world'}`
Invoking: `ConcantenateStrings` with `{'a': 'hello', 'b': 'world'}`

Invoking: `ConcantenateStrings` with `{'a': 'hello', 'b': 'world'}`
Invoking: `ConcantenateStrings` with `{'a': 'hello', 'b': 'world'}`


helloworldThe concatenated result is: **helloworld**.

> Finished chain.
Response for 'Concatenate hello and world': {'input': "Concatenate 'hello' and 'world'", 'output': 'The concatenated result is: **helloworld**.'}
"""
