#!/usr/bin/python3
import os
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from tavily import AsyncTavilyClient
from llama_index.core.agent.workflow import FunctionAgent
import asyncio

#load environment variable
load_dotenv()

#set environment variable
api_key = os.getenv("GROQ_API_KEY")

llm=Groq(model="llama3-70b-8192", api_key=api_key)

#provide the agent with tools to make it more useful
#i.e search the internet using tavily

async def search_web(query:str) -> str:
    """Useful for uisng the web to answer questions"""
    client=AsyncTavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return str(await client.search(query))

#create the Agent Workflow that uses the tool
agent= FunctionAgent(
    tools=[search_web],
    llm=llm,
    system_prompt="You are called Webot, a helpful assistant that can search the web for anything"
)

#running the agent
async def main():
    while True:
        user_query=input("\nYou>> ")
        if user_query.lower() == "exit":
            break
        response=await agent.run(user_msg=user_query)
        print(str(response))

#Entry point
if __name__=="__main__":
    asyncio.run(main())



"""
You>> weather in atlanta 

Webot>> The current weather in Atlanta is partly cloudy with a temperature of 70.0°F (21.1°C) and a wind speed of 4.5 mph (7.2 kph).     

"""