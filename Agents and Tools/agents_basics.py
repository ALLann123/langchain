#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

#load the enviroment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

#define a simple tool function that returns the current time

def get_current_time(*args, **kwargs):
    """Returns the current time n H:MM AM/PM format."""
    import datetime
    now=datetime.datetime.now()  #get current time
    return now.strftime("%I:%M %p")   #format time in H:MM AM/PM format

#list of tools available to the agent
tools = [
    Tool(
        name="Time",   #Name of the tool
        func=get_current_time, #Funtion that the tool will execute
        #description of the tool
        description="Useful when you need to know the current time",
    )
]

#pull the prompt template from the hub
#ReAct = Reason and Action
# https://smith.langchain.com/hub/hwchase17/react

prompt=hub.pull("hwchase17/react")

#initialize the LLM
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

#Create the ReAct agent using the create_react_agent function

agent=create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

#create an agent executor from the agent and tools
agent_executor=AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

#run the agent with a test query
response=agent_executor.invoke({"input": "What time is it?"})

#print the response from the agent
print("response: ", response)
"""
while True:
    print("***************TIME KEEPER AGENT*********************")
    query=input("Own_the_net>> ")
    print("-------------------------------------------------")
    if query.lower() == "exit":
        print("[+]Good Bye!!")
        break
    response=agent_executor.invoke({"input": query})
    print("response: ", response)
    print()

"""

"""
  warnings.warn(
***************TIME KEEPER AGENT*********************
Own_the_net>> time in New York now?
-------------------------------------------------


> Entering new AgentExecutor chain...
I need to use the Time tool to find out the current time in New York.  
Action: Time  
Action Input: Current time in New York (Eastern Time Zone).  12:56 PMI now know the final answer.  
Final Answer: The current time in New York is 12:56 PM.

> Finished chain.
response:  {'input': 'time in New York now?', 'output': 'The current time in New York is 12:56 PM.'}

"""