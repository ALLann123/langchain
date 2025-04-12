#!/usr/bin/python3
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

#expects the tuple format

messages=[
    ("system", "You are a comedian who tells jokes about {topic}"),
    ("human", "Tell me {joke_count} jokes ")
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic":"hackers","joke_count":3})

print("----Prompt with system and human messages--------")
print(prompt)