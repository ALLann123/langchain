#!/usr/bin/python3
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

#load environment variables
load_dotenv()
api_key=os.getenv("GITHUB_TOKEN")

#lets create our chat Model i.e chatgpt
# Create the LangChain chat model using the GitHub Marketplace endpoint
model = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=api_key,
    base_url="https://models.inference.ai.azure.com"
)

#define the prompt templates
prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes ")
    ]
)

#create the combined chain using langchain expression language (LCEL)
chain=prompt_template | model | StrOutputParser()

#run the chain
results=chain.invoke({"topic":"school", "joke_count":3})

#output
print(results)