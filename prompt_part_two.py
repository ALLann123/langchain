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

print("-------Print Prompt with muliple Place holders-------------")
template_multiple="""You are a helpful assistant.
Human: Tell me a {adjective} story about a {animal}
Assisntant"""

prompt_multiple= ChatPromptTemplate.from_template(template_multiple)

prompt= prompt_multiple.invoke({"adjective":"funny", "animal":"snakes"})
#print(prompt)

result=model.invoke(prompt)
print(result.content)

