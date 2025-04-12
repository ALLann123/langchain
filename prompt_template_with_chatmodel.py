#!/usr/bin/python3
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage

#load environment variables
load_dotenv()
api_key=os.getenv("GITHUB_TOKEN")

# Create the LangChain chat model using the GitHub Marketplace endpoint
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

#create chatprompt template
template="Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template)

prompt=prompt_template.invoke({"topic":"cats"})

result=model.invoke(prompt)
print(result.content)
