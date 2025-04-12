#!/usr/bin/python3
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

template_multiple="""You are a helpful assistant.
Human: Tell me a {adjective} story about a {animal}
Assisntant"""

prompt_multiple= ChatPromptTemplate.from_template(template_multiple)

prompt= prompt_multiple.invoke({"adjective":"funny", "animal":"snakes"})
print("-------Print Prompt with muliple Place holders-------------")
print(prompt)
