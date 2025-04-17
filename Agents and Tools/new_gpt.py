#!/usr/bin/python3

import os
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
import datetime

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

# Define tools
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")

# Initialize Wikipedia tool
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# Define tools list
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful when you need to know the current time"
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Useful for searching Wikipedia for general knowledge"
    )
]

# Create memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Add initial system message
memory.chat_memory.add_message(
    SystemMessage(
        content=(
            "You are an AI assistant that can provide helpful answers using available tools. "
            "If you are unable to answer, you can use the Time tool to get the current time "
            "or the Wikipedia tool to search for general knowledge."
        )
    )
)

# Load prompt from LangChain hub
prompt = hub.pull("hwchase17/structured-chat-agent")

# Create agent
agent = create_structured_chat_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
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
        print(f"Error occurred: {str(e)}\n")


"""
You: what is the current time?


> Entering new AgentExecutor chain...
{
  "action": "Time",
  "action_input": {
    "tool_input": ""
  }
}12:18 AM{
  "action": "Final Answer",
  "action_input": "The current time is 12:18 AM."
}

> Finished chain.
Assistant: The current time is 12:18 AM.

You: name 8 tribes in kenya


> Entering new AgentExecutor chain...
{
  "action": "Final Answer",
  "action_input": "Here are 8 tribes in Kenya: Kikuyu, Luo, Luhya, Kalenjin, Kamba, Kisii, Meru, and Maasai."
}

> Finished chain.
Assistant: Here are 8 tribes in Kenya: Kikuyu, Luo, Luhya, Kalenjin, Kamba, Kisii, Meru, and Maasai.

You: Dr. Sylvia Kangara's husband?


> Entering new AgentExecutor chain...
```json
{
  "action": "Final Answer",
  "action_input": "I'm sorry, but I cannot provide personal information about individuals including Dr. Sylvia Kangara's husband."
}
```

> Finished chain.
Assistant: I'm sorry, but I cannot provide personal information about individuals including Dr. Sylvia Kangara's husband.

You: define RAG in relation to AI Agents


> Entering new AgentExecutor chain...
{
  "action": "Final Answer",
  "action_input": "RAG (Retrieval-Augmented Generation) refers to a framework for AI agents that combines natural language generation (NLG) models, like large language models, with retrieval systems. In this setting, the AI uses external knowledge sources (retrieved documents, databases) to augment the generative process for producing responses. This improves accuracy, relevance, and contextual grounding in complex tasks or queries."
}

> Finished chain.
Assistant: RAG (Retrieval-Augmented Generation) refers to a framework for AI agents that combines natural language generation (NLG) models, like large language models, with retrieval systems. In this setting, the AI uses external knowledge sources (retrieved documents, databases) to 
augment the generative process for producing responses. This improves accuracy, relevance, and contextual grounding in complex tasks or queries.
"""