#!/usr/bin/python3
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate


#prompt with system and Human Message(USING Tuples)

messages=[
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes.")
]

prompt_template=ChatPromptTemplate.from_messages(messages)

prompt=prompt_template.invoke({"topic":"lawyers", "joke_count":3})

print("\n-----Prompt with System and Human Message (Tuple)------\n")
print(prompt)

