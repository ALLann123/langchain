from dotenv import load_dotenv
import os
import datetime
import wikipedia

from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Tools
def get_current_time(*args, **kwargs):
    return datetime.datetime.now().strftime("%I:%M %p")

def search_wikipedia(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except Exception as e:
        return f"Error: {str(e)}"

tools = [
    Tool(name="Time", func=get_current_time, description="Get current time."),
    Tool(name="Wikipedia", func=search_wikipedia, description="Search Wikipedia."),
]

# Prompt
prompt = hub.pull("hwchase17/structured-chat-agent")  # Or fallback to a custom one

# Model
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key
)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    handle_parsing_errors=True,
    verbose=True
)

# Set initial context
initial_message = "You are an AI assistant that can use tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat loop
try:
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break

        memory.chat_memory.add_message(HumanMessage(content=user_input))
        response = agent_executor.invoke({"input": user_input})
        print("Bot:", response["output"])
        memory.chat_memory.add_message(AIMessage(content=response["output"]))
except KeyboardInterrupt:
    print("\n[Exiting chat]")
