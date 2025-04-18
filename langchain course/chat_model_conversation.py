#!/usr/bin/python3
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()
api_key = os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

#Now lets set the systemmessage, humanmessage
messages=[
    SystemMessage("You are an expert in social media content strategy"),
    HumanMessage("Give a short tip to create engaging posts on Instagram")
]

result=llm.invoke(messages)

print(result.content)

"""
langchain course>python chat_model_conversation.py
Focus on storytelling! Craft captions that evoke emotion, share relatable experiences, or take your audience behind the scenes. Pair them with eye-catching visuals that align with your message, and always include a clear call-to-action to spark interactions.
"""