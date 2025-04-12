#!/usr/bin/python3
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

template = "Tell me a joke about {topic}"

# Proper way to create a prompt template for chat models
prompt_template = ChatPromptTemplate.from_template(template)

print("-----Prompt from Template------")
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)
