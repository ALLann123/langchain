#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage
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

#lets give the llm the tool above
llm_with_tools=llm.bind_tools([multiply])

tool_call=llm_with_tools.invoke([HumanMessage(content=f"What is 2 multipied by 9")])

print(tool_call)

