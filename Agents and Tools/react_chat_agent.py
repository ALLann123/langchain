#!/usr/bin/python3

import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Initialize LLM
#initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)


# Define tools
def get_current_time(*args, **kwargs):
    """Returns the current time n H:MM AM/PM format."""
    import datetime
    now=datetime.datetime.now()  #get current time
    return now.strftime("%I:%M %p")   #format time in H:MM AM/PM format

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

tools = [
    Tool.from_function(
        name="Time",
        description="Use this tool to get the current time",
        func=get_current_time
    ),
    Tool.from_function(
        name="Wikipedia",
        description="Use this tool to search Wikipedia for general knowledge",
        func=wiki.run
    )
]

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Add initial system message
memory.chat_memory.messages = [
    SystemMessage(
        content=(
            "You are an AI assistant that can provide helpful answers using available tools. "
            "If you are unable to answer, you can use the following tools: Time and Wikipedia."
        )
    )
]

# Load prompt from LangChain hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Create agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Run the agent
print("AI Assistant Ready. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    try:
        response = agent_executor.invoke({"input": user_input})
        print(f"Assistant: {response['output']}\n")
    except Exception as e:
        print(f"Error: {e}\n")

